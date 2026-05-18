""" Util functions for loading and saving checkpoints


"""
import os
import torch


def load_pretrain_checkpoint(model, pretrain_checkpoint_path, logger=None):
    """Load pretrained encoder checkpoint with robust prefix-aware key matching."""
    
    def debug_log(*msgs):
        msg = " ".join(str(m) for m in msgs)
        if logger is not None and hasattr(logger, 'debug'):
            logger.debug(msg)
        elif logger is not None and hasattr(logger, 'fprint'):
            logger.fprint(msg)
        else:
            pass
    
    model_dict = model.state_dict()
    
    # Filter to only encoder keys in the model
    model_encoder_keys = {k: v for k, v in model_dict.items() if k.startswith('encoder.')}
    
    if pretrain_checkpoint_path is None:
        raise ValueError('Pretrained checkpoint must be given.')
    
    debug_log('\n=== CHECKPOINT DIAGNOSTICS ===')
    debug_log(f'Loading encoder module from pretrained checkpoint: {pretrain_checkpoint_path}')
    
    # Load checkpoint on CPU
    checkpoint_path = os.path.join(pretrain_checkpoint_path, 'checkpoint.tar')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate checkpoint format - must contain "params" key
    if 'params' not in checkpoint:
        existing_keys = list(checkpoint.keys())
        raise KeyError(
            f'Checkpoint does not contain "params" key! This is not a valid pretrain checkpoint.\n'
            f'Existing checkpoint keys: {existing_keys}\n'
            'Pretrain checkpoints should contain a "params" key with encoder weights.\n'
            'Meta-training checkpoints contain "model_state_dict", "optimizer_state_dict", etc.\n'
            'Ensure --pretrain_checkpoint_path points to a pretrain checkpoint, not meta-training.'
        )
    
    pretrained_dict = checkpoint['params']
    
    # Statistics
    total_checkpoint_raw_keys = len(pretrained_dict)
    total_model_encoder_keys = len(model_encoder_keys)
    matched_keys = {}
    missing_keys = []
    unexpected_keys = []
    shape_mismatch_keys = []
    matched_key_pairs = []  # Track (raw_key, model_key) pairs for debugging
    
    debug_log(f'\n[Stats] Raw checkpoint keys: {total_checkpoint_raw_keys}')
    debug_log(f'[Stats] Model encoder keys: {total_model_encoder_keys}')
    
    # Print first 50 raw checkpoint keys
    raw_keys_list = list(pretrained_dict.keys())
    debug_log('\n[Raw Checkpoint Keys] First 50:')
    for i, key in enumerate(raw_keys_list[:50], 1):
        debug_log(f'  {i:2d}. {key}')
    if len(raw_keys_list) > 50:
        debug_log(f'  ... and {len(raw_keys_list) - 50} more')
    
    # Print first 50 model encoder keys
    model_keys_list = list(model_encoder_keys.keys())
    debug_log('\n[Model Encoder Keys] First 50:')
    for i, key in enumerate(model_keys_list[:50], 1):
        debug_log(f'  {i:2d}. {key}')
    if len(model_keys_list) > 50:
        debug_log(f'  ... and {len(model_keys_list) - 50} more')
    
    # Prefix-aware key matching - generates candidate model keys for a given checkpoint key
    def get_candidate_keys(k):
        """Generate candidate model keys for a given checkpoint key."""
        candidates = []
        
        # Base candidates: try as-is, with encoder., and with encoder.backbone.
        candidates.append(k)                                  # k
        candidates.append('encoder.' + k)                     # encoder.k
        candidates.append('encoder.backbone.' + k)            # encoder.backbone.k
        
        # Handle module. prefix
        if k.startswith('module.'):
            stripped = k[len('module.'):]                     # k without module.
            candidates.append(stripped)                       # stripped
            candidates.append('encoder.' + stripped)          # encoder.stripped
            candidates.append('encoder.backbone.' + stripped) # encoder.backbone.stripped
        
        # Handle encoder. prefix (already has encoder.)
        if k.startswith('encoder.'):
            stripped = k[len('encoder.'):]                    # k without encoder.
            candidates.append('encoder.backbone.' + stripped) # encoder.backbone.stripped
        
        # Handle module.encoder. prefix
        if k.startswith('module.encoder.'):
            stripped = k[len('module.'):]                     # encoder.xxx
            candidates.append(stripped)                       # encoder.xxx
            stripped2 = k[len('module.encoder.'):]            # xxx
            candidates.append('encoder.backbone.' + stripped2) # encoder.backbone.xxx
        
        # Handle backbone. prefix
        if k.startswith('backbone.'):
            candidates.append('encoder.' + k)                  # encoder.backbone.xxx
            stripped = k[len('backbone.'):]                   # xxx
            candidates.append('encoder.' + stripped)           # encoder.xxx
            candidates.append('encoder.backbone.' + stripped)  # encoder.backbone.xxx
        
        # Handle module.backbone. prefix
        if k.startswith('module.backbone.'):
            stripped = k[len('module.'):]                     # backbone.xxx
            candidates.append('encoder.' + stripped)          # encoder.backbone.xxx
            stripped2 = k[len('module.backbone.'):]           # xxx
            candidates.append('encoder.' + stripped2)         # encoder.xxx
            candidates.append('encoder.backbone.' + stripped2) # encoder.backbone.xxx
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates
    
    # Try to match each checkpoint key to model
    for ckpt_key, ckpt_value in pretrained_dict.items():
        candidates = get_candidate_keys(ckpt_key)
        matched = False
        
        for candidate in candidates:
            if candidate in model_dict:
                model_value = model_dict[candidate]
                if ckpt_value.shape == model_value.shape:
                    # Only add if not already matched (avoid duplicates)
                    if candidate not in matched_keys:
                        matched_keys[candidate] = ckpt_value
                        matched_key_pairs.append((ckpt_key, candidate))
                    matched = True
                    break
                else:
                    shape_mismatch_keys.append(
                        f'{ckpt_key} -> {candidate}: checkpoint_shape={ckpt_value.shape}, model_shape={model_value.shape}'
                    )
        
        if not matched:
            unexpected_keys.append(ckpt_key)
    
    # Check for missing keys (model encoder keys not matched)
    for model_key in model_encoder_keys:
        if model_key not in matched_keys:
            missing_keys.append(model_key)
    
    # Print matched key pairs
    debug_log(f'\n[Matched Keys] {len(matched_key_pairs)} matched:')
    if matched_key_pairs:
        debug_log('  Raw key -> Model key:')
        for i, (raw_key, matched_model_key) in enumerate(matched_key_pairs[:50], 1):
            debug_log(f'  {i:2d}. {raw_key} -> {matched_model_key}')
        if len(matched_key_pairs) > 50:
            debug_log(f'  ... and {len(matched_key_pairs) - 50} more')
    
    # Print missing keys
    if missing_keys:
        debug_log(f'\n[Missing Keys] {len(missing_keys)} model encoder keys not found in checkpoint:')
        for i, key in enumerate(missing_keys[:50], 1):
            debug_log(f'  {i:2d}. {key}')
        if len(missing_keys) > 50:
            debug_log(f'  ... and {len(missing_keys) - 50} more')
    
    # Print unexpected keys
    if unexpected_keys:
        debug_log(f'\n[Unexpected Keys] {len(unexpected_keys)} checkpoint keys could not be matched:')
        for i, key in enumerate(unexpected_keys[:50], 1):
            debug_log(f'  {i:2d}. {key}')
        if len(unexpected_keys) > 50:
            debug_log(f'  ... and {len(unexpected_keys) - 50} more')
    
    # Print shape mismatches
    if shape_mismatch_keys:
        debug_log(f'\n[Shape Mismatches] {len(shape_mismatch_keys)} keys with shape mismatch:')
        for i, key_info in enumerate(shape_mismatch_keys[:50], 1):
            debug_log(f'  {i:2d}. {key_info}')
        if len(shape_mismatch_keys) > 50:
            debug_log(f'  ... and {len(shape_mismatch_keys) - 50} more')
    
    # Calculate loading ratio
    loaded_ratio = len(matched_keys) / total_model_encoder_keys if total_model_encoder_keys > 0 else 0.0
    debug_log(f'\n[Loading Ratio] {loaded_ratio:.4f} ({len(matched_keys)}/{total_model_encoder_keys})')
    
    # Only load if ratio passes threshold
    if loaded_ratio >= 0.90:
        model_dict.update(matched_keys)
        model.load_state_dict(model_dict, strict=False)
        debug_log('\n[SUCCESS] Pretrained encoder weights loaded successfully!')
    else:
        raise RuntimeError(
            f'[ERROR] Checkpoint loading ratio ({loaded_ratio:.4f}) is below 0.90!\n'
            'This indicates a mismatch between the pretrained checkpoint and current model.\n'
            'Please verify:\n'
            '  1. --use_high_dgcnn setting matches pretraining\n'
            '  2. --backbone_name setting matches pretraining\n'
            '  3. --pretrain_checkpoint_path points to correct pretrain checkpoint\n'
            '  4. Ensure pretrained encoder architecture matches current model\n'
            'See diagnostic output above for key matching details.'
        )
    
    debug_log('=== END CHECKPOINT DIAGNOSTICS ===\n')
    return model


def load_model_checkpoint(model, model_checkpoint_path, optimizer=None, mode='test', logger=None):
    def debug_log(*msgs):
        msg = " ".join(str(m) for m in msgs)
        if logger is not None and hasattr(logger, 'debug'):
            logger.debug(msg)
        elif logger is not None and hasattr(logger, 'fprint'):
            logger.fprint(msg)
        else:
            pass

    try:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, 'checkpoint.tar'))
        start_iter = checkpoint['iteration']
        start_iou = checkpoint['IoU']
    except:
        raise ValueError('Model checkpoint file must be correctly given (%s).' %model_checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if mode == 'test':
        debug_log('Load model checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model
    else:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            debug_log('Checkpoint does not include optimizer state dict...')
        debug_log('Resume from checkpoint at Iteration %d (IoU %f)...' % (start_iter, start_iou))
        return model, optimizer


def save_pretrain_checkpoint(model, output_path):
    torch.save(dict(params=model.encoder.state_dict()), os.path.join(output_path, 'checkpoint.tar'))