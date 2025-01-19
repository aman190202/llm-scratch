from torch.utils.data import Dataset, DataLoader
import torch
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader





def readText(fileName = "the-verdict.txt"):
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text

if __name__=='__main__':

    # Step 1 : Import Raw Data
    raw_text = readText("the-verdict.txt")

    # Step 2 : Convert Raw Data to embeddings
    
    # Context Length
    max_length = 3
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print("Token IDs:\n", inputs)
   #  print("\nInputs shape:\\n", inputs.shape)


    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings

    # print(input_embeddings.shape) # A batch of 8 inputs, with length each of 4 ( Eg : " I am a good"), converted into an embedding of 256 dimensions - [8,4,256]


    # Attention part begins
