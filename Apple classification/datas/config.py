#config.py
cfg  = {
    'image_size':(224,224),
    'batch_size': 8,
    'learning_rate' : 0.001,
    'weight_decay' : 1e-4,
    'epochs':50,
    'patience':50,
    'step_size':5,
    'gamma':0.1,
    'dropout' : 0.5,
    "dataset": "appledataset"
}