class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Add the first layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Add hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Add the output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
    def forward(self, x):
        # Apply ReLU to all layers except the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Output layer without ReLU
        x = self.layers[-1](x)
        return x


class UserAttributeEmbeddings(nn.Module):
    def __init__(self, embedding_fn):
        super(UserAttributeEmbeddings, self).__init__()
        self.embedding_userGender = embedding_fn['userGender']
        self.embedding_userAgeIds = embedding_fn['userAgeIds']
        self.embedding_userPrefectureIds =  embedding_fn['userPrefectureIds']
        self.output_dim = self.embedding_userGender.embedding_dim + self.embedding_userAgeIds.embedding_dim + self.embedding_userPrefectureIds.embedding_dim
    def forward(self, userGender, userAgeIds , userPrefectureIds):
        embedded_userGender = self.embedding_userGender(userGender).squeeze(dim=0)
        embedded_userAgeIds = self.embedding_userAgeIds(userAgeIds).squeeze(dim=0)
        embedded_userPrefectureIds = self.embedding_userPrefectureIds(userPrefectureIds).squeeze(dim=0)
        # Concatenate embeddings along the last dimension
        # print('!!!embed shape userAgeIds', embedded_userAgeIds.shape)
        return torch.cat((embedded_userGender, embedded_userAgeIds, embedded_userPrefectureIds), dim=-1)

# image user sim model
class ImageUserSimilarityModel(nn.Module):
    def __init__(self, image_feature_size, embedding_fn, hidden_size, output_size, user_n_layer=2, image_n_layer=2):
        super(ImageUserSimilarityModel, self).__init__()
        self.image_mlp = MLP(image_feature_size, hidden_size, output_size, n_layers=image_n_layer )
        self.user_embeddings = UserAttributeEmbeddings(embedding_fn)
        # Adjust the MLP input size to three times the embedding size (for three concatenated embeddings)
        self.user_mlp = MLP(self.user_embeddings.output_dim, hidden_size, output_size, n_layers=user_n_layer)

    def forward(self, image_features,  userGender, userAgeIds , userPrefectureIds):
        # Process image features, expected shape should be (bs, 768)
        image_output = self.image_mlp(image_features)

        # Process user attributes, expected shape should be (bs, 1)
        user_embedded = self.user_embeddings(userGender, userAgeIds , userPrefectureIds)
        # print('user_embed dim', user_embedded.shape)
        user_output = self.user_mlp(user_embedded).squeeze(1)
        # print('image repr dim', image_output.shape, 'user repr dim', user_output.shape)
        
        # Normalize the outputs
        # print(image_output.shape, user_output.shape)
        image_output = F.normalize(image_output, p=2, dim=1)
        user_output = F.normalize(user_output, p=2, dim=1)

        # Compute cosine similarity
        similarity = (image_output * user_output).sum(dim=1)
        return similarity
