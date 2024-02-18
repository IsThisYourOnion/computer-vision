config = [
    {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': {'pool_size': 2, 'stride': 2}},
    {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': {'pool_size': 2, 'stride': 2}},
    {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': {'pool_size': 2, 'stride': 2}}
]

input_dim = (32, 32)

num_classes=10