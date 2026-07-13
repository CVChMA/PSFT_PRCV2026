import timm



def Uni3d_B(config, args): 
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(
    'eva02_base_patch14_448',
    pretrained=False,
    drop_path_rate=config.drop_path_rate,
    num_classes=0  
    )
    
    # create whole point cloud encoder
    if args.train_mode == 'FT':
        from .point_encoder import PointcloudEncoder
        point_encoder = PointcloudEncoder(point_transformer, config, args)
    elif args.train_mode == 'PG':
        from .point_encoder_PG import PointcloudEncoder
        point_encoder = PointcloudEncoder(point_transformer, config, args)
    return point_encoder
