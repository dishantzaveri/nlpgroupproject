from transformers.models.llava_next.modeling_llava_next import LlavaNextAttention
from .attention import LlavaNextAttention_CaM

def convert_llava_cam(model, config):
    """Recursively replace attention layers with CaM-enabled versions"""
    for name, module in model.named_children():
        if isinstance(module, LlavaNextAttention):
            # Preserve original weights while replacing class
            new_attn = LlavaNextAttention_CaM(config)
            new_attn.load_state_dict(module.state_dict(), strict=True)
            setattr(model, name, new_attn)
        elif len(list(module.children())) > 0:
            # Recursive call for nested modules
            convert_llava_cam(module, config)
    return model