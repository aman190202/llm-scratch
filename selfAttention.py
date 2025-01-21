import torch




inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# Dot product between each vector tells us how similar they are
attention_scores = inputs @ inputs.T
attention_weights = torch.softmax(attention_scores, dim=-1)
all_context_vecs = attention_weights @ inputs # defined by z


# scaled dot product attention -- introduction of weight matrices that are trainable
# w_q, w_k, w_v - query, key and value vector


x_2 = inputs[1]
d_in = inputs.shape[1] # input embedding size
d_out = 2 # usually the same with the input embedding side


# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# query_2 = x_2 @ W_query 
# key_2 = x_2 @ W_key 
# value_2 = x_2 @ W_value

# keys = inputs @ W_key
# values = inputs @ W_value

from main import SelfAttention_v1, SelfAttention_v2
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# the reason for normalization is to improve the training performace by avoiding small gradients


# Causal Attention (masked attention) - hiding the future

sa_v2 = SelfAttention_v2(d_in, d_out)
queries = sa_v2.W_keys(inputs)
keys = sav