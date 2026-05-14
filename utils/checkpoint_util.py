""" Util functions for loading and saving checkpoints


"""
import os
import torch


def load_pretrain_checkpoint(model, pretrain_checkpoint_path):
    # load pretrained model for point cloud encoding
    model_dict = model.state_dict()
    
    # Filter to only encoder keys in the model
    model_encoder_keys = {k: v for k, v in model_dict.items() if k.startswith('encoder.')}
    
    if pretrain_checkpoint_path is not None:
        print('\n=== CHECKPOINT DIAGNOSTICS ===')
        print(f'Loading encoder module from pretrained checkpoint: {pretrain_checkpoint_path}')
        
        # Load checkpoint on CPU to avoid GPU memory issues
        checkpoint = torch.load(os.path.join(pretrain_checkpoint_path, 'checkpoint.tar'), map_location='cpu')
        pretrained_dict = checkpoint['params']
        
        # Convert checkpoint keys to model format by prefixing 'encoder.'
        pretrained_encoder_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        
        # Statistics
        total_checkpoint_keys = len(pretrained_encoder_dict)
        total_model_encoder_keys = len(model_encoder_keys)
        matched_keys = []
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []
        
        print(f'\nCheckpoint encoder keys: {total_checkpoint_keys}')
        print(f'Model encoder keys: {total_model_encoder_keys}')
        
        # Check each checkpoint key
        for ckpt_key, ckpt_value in pretrained_encoder_dict.items():
            if ckpt_key not in model_dict:
                unexpected_keys.append(ckpt_key)
            else:
                model_value = model_dict[ckpt_key]
                if ckpt_value.shape == model_value.shape:
                    matched_keys.append(ckpt_key)
                else:
                    shape_mismatch_keys.append(
                        f'{ckpt_key}: checkpoint_shape={ckpt_value.shape}, model_shape={model_value.shape}'
                    )
        
        # Check for missing keys (model keys not in checkpoint)
        for model_key in model_encoder_keys:
            if model_key not in pretrained_encoder_dict:
                missing_keys.append(model_key)
        
        # Print diagnostics
        print(f'\nSuccessfully matched encoder keys: {len(matched_keys)}')
        
        if missing_keys:
            print(f'\nMissing encoder keys ({len(missing_keys)}):')
            for key in missing_keys[:30]:
                print(f'  - {key}')
            if len(missing_keys) > 30:
                print(f'  ... and {len(missing_keys) - 30} more')
        
        if unexpected_keys:
            print(f'\nUnexpected checkpoint keys ({len(unexpected_keys)}):')
            for key in unexpected_keys[:30]:
                print(f'  - {key}')
            if len(unexpected_keys) > 30:
                print(f'  ... and {len(unexpected_keys) - 30} more')
        
        if shape_mismatch_keys:
            print(f'\nShape mismatch keys ({len(shape_mismatch_keys)}):')
            for key_info in shape_mismatch_keys[:30]:
                print(f'  - {key_info}')
            if len(shape_mismatch_keys) > 30:
                print(f'  ... and {len(shape_mismatch_keys) - 30} more')
        
        # Calculate loading ratio
        loaded_ratio = len(matched_keys) / total_model_encoder_keys if total_model_encoder_keys > 0 else 0.0
        print(f'\nLoaded ratio: {loaded_ratio:.4f} ({len(matched_keys)}/{total_model_encoder_keys})')
        
        # Only load parameters whose key exists in model and shape matches
        filtered_pretrained_dict = {
            k: v for k, v in pretrained_encoder_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        model_dict.update(filtered_pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # Raise error if loading ratio is too low
        if loaded_ratio < 0.90:
            raise RuntimeError(
                f'Checkpoint loading ratio ({loaded_ratio:.4f}) is below 0.90!\n'
                'This indicates a mismatch between the pretrained checkpoint and current model.\n'
                'Please check: --use_high_dgcnn, backbone_name, dataset fold, and checkpoint path.'
            )
        
        print('=== END CHECKPOINT DIAGNOSTICS ===\n')
    else:
        raise ValueError('Pretrained checkpoint must be given.')

    return model


def load_model_checkpoint(model, model_checkpoint_path, optimizer=None, mode='test'):
    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.tar'))
        start_iter = checkpoint['iteration']
        start_iou = checkpoint['IoU']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if mode == 'test':
        print('Load model checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model
    else:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print('Checkpoint does not include optimizer state dict...')
        print('Resume from checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model, optimizer


def save_pretrain_checkpoint(model, output_path):
    torch.save(dict(params=model.encoder.state_dict()), os.path.join(output_path, 'checkpoint.tar'))