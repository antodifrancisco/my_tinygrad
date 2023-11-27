from tinygrad import Tensor, nn

"""
model.layers.0.self_attn.q_proj.weight (4096, 4096)
model.layers.0.self_attn.k_proj.weight (1024, 4096)
model.layers.0.self_attn.v_proj.weight (1024, 4096)
model.layers.0.self_attn.o_proj.weight (4096, 4096)
model.layers.0.mlp.gate_proj.weight (14336, 4096)
model.layers.0.mlp.up_proj.weight (14336, 4096)
model.layers.0.mlp.down_proj.weight (4096, 14336)
model.layers.0.input_layernorm.weight (4096,)
model.layers.0.post_attention_layernorm.weight (4096,)

"""

if __name__ == "__main__":
  Tensor.no_grad = True
  # TODO: add read only Tensors
  part1 = nn.state.torch_load("../weights/OpenHermes/pytorch_model-00001-of-00002.bin")
  part2 = nn.state.torch_load("../weights/OpenHermes/pytorch_model-00002-of-00002.bin")

  for k,v in part1.items():
    print(k,v.shape)