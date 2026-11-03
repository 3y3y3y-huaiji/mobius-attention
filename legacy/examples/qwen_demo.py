r"""
Minimal Example: Injecting Semantic Gauge Möbius Attention into Qwen
Dual-Licensed under Apache-2.0 and MulanPSL-2.0.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.mobius_attention import SemanticGaugeConnection, inject_mobius_attention

def main():
    model_id = "Qwen/Qwen3.5-9B"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 1. Non-invasively patch rotary embedding modules with Möbius spinor operator
    count = inject_mobius_attention(model)
    print(f"Successfully patched {count} rotary embedding layers with Möbius operator.")
    
    # 2. Build semantic gauge connection from input embeddings
    embed = model.get_input_embeddings()
    gauge = SemanticGaugeConnection(hidden_dim=embed.embedding_dim).to(model.device)
    
    # 3. Test prompt with arbitrary logic inversion
    prompt = (
        "Directive: FACT_ALPHA is currently ACTIVE and VALID. "
        "Maintain routine verification across localized subnets. "
        "CRITICAL NOTICE: However, the prior directive is INVALIDATED and FALSE. "
        "Query: Determine the final truth state of FACT_ALPHA: "
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        embs = embed(inputs.input_ids).float()
        omega, phi = gauge(embs)
        
        # Propagate dynamic holonomy phase into the patched rotary embeddings
        for module in model.modules():
            if hasattr(module, "set_holonomy_phase"):
                module.set_holonomy_phase(phi[0])
                
        outputs = model.generate(**inputs, max_new_tokens=64)
        print("\nResponse with Möbius Attention:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
