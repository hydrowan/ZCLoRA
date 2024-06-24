import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ZCLoRALinear(nn.Module):
    """
    LoRA class that utilises zero initialised convolutions
    This means that the LoRA does not degrade the model at all initially.
    The loss landscape is not as simple as standard LoRA.

    This idea was built to fine tune LLMs with very limited data, when
    catastrophic interference was not an issue, for a hackathon project.

    This class is designed to replace Linear layers in Mistral, may
    require modification for specific use-cases.
    """
    def __init__(self, original_layer, rank=4, alpha=32, dtype=torch.float32):
        """
        Params:
            original_layer (object)
                Original Linear layer objet to be replaced
                Assumes a specific structure including original_layer.weight existing
                as a torch weight matrix etc.
            
            rank (int)
            alpha (int)
                Standard LoRA parameters, default values here are aggressive

        """

        super().__init__()
        # Externally set inference variables
        self.apply_lora = True 
        self.lora_strength = 1.0 
        # Use apply_lora = False over lora_strength = 0.0 as it saves compute

        device = original_layer.weight.device
        self.W = original_layer.weight.clone().detach()
        self.W.requires_grad = False
        self.rank = rank

        self.dtype = dtype
        self.original_dtype = self.W.dtype

        in_features, out_features = original_layer.weight.shape
        self.A = nn.Parameter(torch.randn((in_features, rank), device=device, dtype=self.dtype))
        self.B = nn.Parameter(torch.randn((rank, out_features), device=device, dtype=self.dtype))

        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

        self.scaling = alpha / rank
        self.zero_conv = nn.Conv2d(1,1,kernel_size=1, bias=True, device=device, dtype=self.dtype)
        self.zero_conv.weight.data.fill_(0)
        self.zero_conv.bias.data.fill_(0)

    def forward(self, x):
        x_dtype = x.dtype # x and self.dtype are likely f32 anyway
        x = x.to(self.dtype)

        if self.apply_lora is False:
            x = torch.matmul(x, self.W.T.to(self.dtype)) + (self.bias.to(self.dtype) if self.bias is not None else 0)
            return x.to(x_dtype)

        low_rank_update = self.scaling * torch.matmul(self.A, self.B)

        dim = len(low_rank_update.shape)

        if dim == 2: #unbatched
            low_rank_update = rearrange(
                self.zero_conv(
                    rearrange(
                        low_rank_update,
                        "h w -> 1 h w"
                    )
                ),
                "1 h w -> h w"
            )
            

        elif dim == 3: #batched
            low_rank_update = rearrange(
                self.zero_conv(
                    rearrange(
                        low_rank_update,
                        "b h w -> b 1 h w"
                    )
                ),
                "b 1 h w -> b h w"
            )

        W_prime = self.lora_strength * self.W.to(self.dtype) + low_rank_update

        x = torch.matmul(x, W_prime.T) + (self.bias.to(self.dtype) if self.bias is not None else 0)
        return x.to(x_dtype)


    @staticmethod
    def apply_model(pipe, rank, alpha):
        """
        Rough example usage, example for transformers pipeline instance of Mistral.
        """

        print("Disabling grad for all model params")
        for param in pipe.model.parameters():
            param.requires_grad = False

        rank = 4
        alpha = 32

        trainable_layers = []

        print("Adapting all self attn matrices and linear projections to ZCLoRA")
        for layer_idx in range(0,32):
            layer = pipe.model._modules['model'].layers[layer_idx].self_attn
            layer.q_proj = ZCLoRALinear(layer.q_proj, rank=rank, alpha=alpha)
            layer.k_proj = ZCLoRALinear(layer.k_proj, rank=rank, alpha=alpha)
            layer.v_proj = ZCLoRALinear(layer.v_proj, rank=rank, alpha=alpha)
            layer.o_proj = ZCLoRALinear(layer.o_proj, rank=rank, alpha=alpha)
            trainable_layers.extend([layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj])

        print("Confirming grad on only ZCLoRA")
        for name, param in pipe.model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
        
        return pipe, trainable_layers

if __name__ == "__main__":
    import unittest

    class TestLoRA(unittest.TestCase):
        def setUp(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.original_layer = nn.Linear(1024,1024,bias=False,dtype=torch.bfloat16,device=self.device)
            self.vector = torch.randn(1024, device=self.device, dtype=torch.float32)
            self.vector.requires_grad = True
        
        def test_grad_ZC(self):
            """
            Tests gradients flow and ZeroConv inhibits LoRA effect
            """
            linear = ZCLoRALinear(original_layer=self.original_layer)
            output = linear(self.vector)
            output.sum().backward()
            self.assertIsNotNone(self.vector.grad, "Gradient should be computed")
            self.assertTrue(torch.any(self.vector.grad != 0), "Gradient should not be zero")

            linear.apply_lora = False
            output2 = linear(self.vector)
            output2.sum().backward()
            self.assertIsNotNone(self.vector.grad, "Gradient should be computed")
            self.assertTrue(torch.any(self.vector.grad != 0), "Gradient should not be zero")

            self.assertEqual(output.sum(),output2.sum(), "LoRA should have no effect without training")
        
    
    unittest.main()
        
