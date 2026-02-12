import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import warnings


class TimeseriesForecasting(nn.Module):
    def __init__(self, rnn_cell=nn.LSTM, input_size=4, hidden_size=150, num_layers=1, output_size=2,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super(TimeseriesForecasting, self).__init__()
        
        # Input and Recurrent Cell
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.rnn_cell = rnn_cell(
            input_size=input_size, 
            num_layers=self.num_layers, 
            hidden_size=self.hidden_size, 
            batch_first=self.batch_first, 
            bidirectional=self.bidirectional, 
            **kwargs
        )

        self.fc_layer = lambda in_feats, out_feats: nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=out_feats),
            nn.ReLU(),
        )

        # Output layers
        self.output_size = output_size
        fc_layers = [2 * hidden_size if self.bidirectional else hidden_size, *fc_layers, self.output_size]
        self.fc = nn.Sequential(
            *[self.fc_layer(in_feats, out_feats) for in_feats, out_feats in zip(fc_layers, fc_layers[1:-1])],
            nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1])
        )
                        
        self.scale = scale
        if self.scale is not None:
            self.mu, self.sigma = self.scale['mu'], self.scale['sigma']    
        else:
            self.mu, self.sigma = torch.zeros(1,), torch.ones(1,)
            warnings.warn("Instantiated instance without standardization. Falling back to identity function...")

    def forward_rnn_cell(self, x, lengths):
        # Sort input sequences by length in descending order
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        sorted_x = x[sorted_idx]

        # Pack the sorted sequences
        packed_x = pack_padded_sequence(sorted_x, sorted_lengths.cpu(), batch_first=True)

        # Initialize ```hidden state``` and ```cell state``` with zeros
        h0 = torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
    
        # Forward propagate packed sequences through GRU 
        if isinstance(self.rnn_cell, nn.GRU):
            packed_out, h_n = self.rnn_cell(packed_x, h0)
            
            # Reorder the output sequences to match the original input order
            _, reversed_idx = sorted_idx.sort(0)
            return packed_out, (h_n, None), sorted_idx, reversed_idx
        
        # Forward propagate packed sequences through LSTM
        c0 = torch.zeros(2*self.num_layers if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        packed_out, (h_n, c_n) = self.rnn_cell(packed_x, (h0, c0))
        
        # Reorder the output sequences to match the original input order
        _, reversed_idx = sorted_idx.sort(0)
        return packed_out, (h_n, c_n), sorted_idx, reversed_idx
    

    def forward(self, x, lengths):
        # Initialize ```hidden state``` and ```cell state``` with zeros
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        packed_out, (h_n, c_n), _, ix = self.forward_rnn_cell(x, lengths)

        # Unpack the output sequences
        # out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Decode the hidden state of the last time step
        out = self.fc(
                torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1]
        )     
        
        # return torch.add(torch.mul(out[ix], self.sigma), self.mu)
        return torch.add(torch.mul(out[ix], self.sigma.tile(self.output_size // 2)), self.mu.tile(self.output_size // 2))


class EnergyDemandForecasting_v2(TimeseriesForecasting):
    def __init__(self, location_embeddings, model_embeddings, rnn_cell=nn.LSTM, input_size=4, misc_size=4, hidden_size=150, num_layers=1, output_size=1,
                 batch_first=True, fc_layers=[50,], scale=None, bidirectional=False, **kwargs):
        super().__init__(
            rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,
            batch_first=batch_first, fc_layers=fc_layers, scale=scale, bidirectional=bidirectional, **kwargs
        )

        # Output layers
        fc_layers = [(
                2 * hidden_size if self.bidirectional else hidden_size
            ) + misc_size + sum(
                [embedding.embedding_dim for embedding in [location_embeddings, model_embeddings]]
            ), *fc_layers, self.output_size
        ]

        self.fc = nn.Sequential(
            *[self.fc_layer(in_feats, out_feats) for in_feats, out_feats in zip(fc_layers, fc_layers[1:-1])],
            nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1])
        )

        # Embedding (convert/cluster shiptypes)
        self.location_embeddings = location_embeddings   
        self.model_embeddings = model_embeddings   
        self.dropout = nn.Dropout(0.13)


    def forward(self, x, lengths, locations, ev_model, *args):
        self.mu, self.sigma = self.mu.to(x.device), self.sigma.to(x.device)

        # Sort input sequences by length in descending order
        _, (h_n, _), ix_sort, ix_rev = self.forward_rnn_cell(x, lengths)

        location_embedding = self.location_embeddings(locations[ix_sort])
        model_embedding = self.model_embeddings(ev_model[ix_sort])

        # Decode the hidden state of the last time step
        out = self.fc(
            self.dropout(
                torch.cat(
                    (
                        torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) if self.bidirectional else h_n[-1, :, :],
                        location_embedding,
                        model_embedding,
                        *args
                    ), 
                    dim=-1
                )
            )
        ) 
        
        return torch.add(
            torch.mul(
                out[ix_rev], 
                self.sigma.tile(self.output_size)
            ), 
            self.mu.tile(self.output_size)
        ).unsqueeze(-1)


# Define the FCBlock class with Kaiming initialization
class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_feats, out_features=out_feats)
        self.relu = nn.ReLU()

        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.fc.weight, mode="fan_in", nonlinearity="relu")

        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.relu(self.fc(x))


class FedXGBllrCNN(nn.Module):
    def __init__(
            self,
            num_clients: int,
            trees_per_client: int,
            in_channels: int = 1,
            conv_channels: int = 64,
            fc_layers: list = [],
            out_channels: int = 1,
            dropout_rate: float = 0.5
        ):
        """
        Initializes the one-layer 1D CNN for FedXGBllr.

        Args:
            num_clients (int): K, the total number of clients participating.
            trees_per_client (int): M, the number of trees in each client's ensemble.
                                    (Note: The source's general setup uses 500 trees total divided by clients,
                                    but your query specifies 1000 estimators *each*.) [6]
            num_conv_channels (int): The number of output channels for the 1D convolution layer. [3]
        """
        super(FedXGBllrCNN, self).__init__()

        # Calculate the total number of trees across all aggregated ensembles [7]
        self.total_trees = num_clients * trees_per_client

        # The kernel size and stride of the 1D convolution are equal to
        # the number of trees (M) in each client's tree ensemble. [2]
        self.kernel_size = trees_per_client
        self.stride = trees_per_client

        # The 1D convolution layer with 1 input channel (for prediction outcomes) [2]
        # and a specified number of output channels.
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,        # Input: prediction outcomes of all trees (treated as a sequence) [2]
            out_channels=conv_channels,     # Number of learning rate strategies [3, 4]
            kernel_size=self.kernel_size,   # Equal to M (trees_per_client) [2]
            stride=self.stride,             # Equal to M (trees_per_client) [2]
            padding=0
        )
        # self.gn1 = nn.GroupNorm(num_groups=conv_channels//2, num_channels=conv_channels)  # e.g., 32 groups of 2 channels for **64** conv. channels
        
        # Activation function G, set to ReLU to avoid overfitting. [5]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Adjust dropout rate as needed

        # Calculate the flattened dimension for the Fully Connected (FC) layer [4]
        conv_output_length = (self.total_trees - self.kernel_size) // self.stride + 1
        flattened_dimension = conv_channels * conv_output_length

        # Output layers
        fc_layers, fc_blocks = [flattened_dimension, *fc_layers, out_channels], []
        for in_feats, out_feats in zip(fc_layers[:-2], fc_layers[1:-1]):
            fc_blocks.append(FCBlock(in_feats, out_feats))
        
        # Add the final linear layer without ReLU activation
        fc_blocks.append(nn.Linear(in_features=fc_layers[-2], out_features=fc_layers[-1]))
        self.fc = nn.Sequential(*fc_blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 1D CNN.

        Args:
            x (torch.Tensor): Input tensor of prediction outcomes from all trees.
                              Expected shape: (batch_size, 1, total_trees)

        Returns:
            torch.Tensor: The final predicted output.
        """
        # Apply 1D convolution
        x = self.conv1d(x)  # Shape: (batch_size, num_conv_channels, conv_output_length)
        
        # # Apply GroupNorm
        # x = self.gn1(x)  # Normalize across groups of channels
        
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, start_dim=1) # Shape: (batch_size, flattened_dimension)
        
        # Apply ReLU activation
        x = self.relu(x)    # Shape: (batch_size, num_conv_channels, conv_output_length)
        
        # Apply Dropout
        x = self.dropout(x)

        # Apply the fully connected layer to get the final prediction
        x = self.fc(x)      # Shape: (batch_size, 1)

        return x
