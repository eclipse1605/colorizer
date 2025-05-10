import torch 
import torch .nn as nn 
import torch .nn .functional as F 


class ConvBlock (nn .Module ):
    def __init__ (self ,in_channels ,out_channels ,kernel_size =3 ,padding =1 ,use_bn =True ,use_relu =True ):
        super (ConvBlock ,self ).__init__ ()

        layers =[]
        layers .append (nn .Conv2d (in_channels ,out_channels ,kernel_size =kernel_size ,padding =padding ))

        if use_bn :
            layers .append (nn .BatchNorm2d (out_channels ))

        if use_relu :
            layers .append (nn .ReLU (inplace =True ))

        self .block =nn .Sequential (*layers )

    def forward (self ,x ):
        return self .block (x )


class DownsampleBlock (nn .Module ):
    def __init__ (self ,in_channels ,out_channels ,kernel_size =3 ,padding =1 ):
        super (DownsampleBlock ,self ).__init__ ()

        self .conv1 =ConvBlock (in_channels ,out_channels ,kernel_size ,padding )
        self .conv2 =ConvBlock (out_channels ,out_channels ,kernel_size ,padding )
        self .pool =nn .MaxPool2d (kernel_size =2 ,stride =2 )

    def forward (self ,x ):

        x =self .conv1 (x )
        x =self .conv2 (x )

        skip =x 

        x =self .pool (x )

        return x ,skip 


class UpsampleBlock (nn .Module ):
    def __init__ (self ,in_channels ,out_channels ,kernel_size =3 ,stride =1 ,padding =1 ):
        super (UpsampleBlock ,self ).__init__ ()
        self .conv =nn .Conv2d (in_channels ,out_channels ,kernel_size =kernel_size ,
        stride =stride ,padding =padding )
        self .bn =nn .BatchNorm2d (out_channels )

    def forward (self ,x ,skip_connection =None ):
        x =F .interpolate (x ,scale_factor =2 ,mode ='nearest')
        if skip_connection is not None :
            x =torch .cat ([x ,skip_connection ],dim =1 )
        x =self .conv (x )
        x =self .bn (x )
        x =F .relu (x )
        return x 


class ECCVColorizer (nn .Module ):
    """Implementation inspired by Zhang et al. ECCV 2016 paper: Colorful Image Colorization"""
    def __init__ (self ):
        super (ECCVColorizer ,self ).__init__ ()


        self .conv1 =nn .Conv2d (1 ,64 ,kernel_size =3 ,stride =1 ,padding =1 )
        self .conv2 =nn .Conv2d (64 ,64 ,kernel_size =3 ,stride =2 ,padding =1 )
        self .conv3 =nn .Conv2d (64 ,128 ,kernel_size =3 ,stride =1 ,padding =1 )
        self .conv4 =nn .Conv2d (128 ,128 ,kernel_size =3 ,stride =2 ,padding =1 )
        self .conv5 =nn .Conv2d (128 ,256 ,kernel_size =3 ,stride =1 ,padding =1 )
        self .conv6 =nn .Conv2d (256 ,256 ,kernel_size =3 ,stride =2 ,padding =1 )


        self .conv7 =nn .Conv2d (256 ,512 ,kernel_size =3 ,stride =1 ,padding =1 )
        self .conv8 =nn .Conv2d (512 ,512 ,kernel_size =3 ,stride =1 ,padding =1 )
        self .conv9 =nn .Conv2d (512 ,256 ,kernel_size =3 ,stride =1 ,padding =1 )


        self .upsample1 =UpsampleBlock (256 ,128 )
        self .upsample2 =UpsampleBlock (128 ,64 )
        self .upsample3 =UpsampleBlock (64 ,32 )


        self .output =nn .Conv2d (32 ,2 ,kernel_size =1 ,stride =1 ,padding =0 )


        self .bn1 =nn .BatchNorm2d (64 )
        self .bn2 =nn .BatchNorm2d (64 )
        self .bn3 =nn .BatchNorm2d (128 )
        self .bn4 =nn .BatchNorm2d (128 )
        self .bn5 =nn .BatchNorm2d (256 )
        self .bn6 =nn .BatchNorm2d (256 )
        self .bn7 =nn .BatchNorm2d (512 )
        self .bn8 =nn .BatchNorm2d (512 )
        self .bn9 =nn .BatchNorm2d (256 )

    def forward (self ,x ):

        x =F .relu (self .bn1 (self .conv1 (x )))
        x =F .relu (self .bn2 (self .conv2 (x )))
        x =F .relu (self .bn3 (self .conv3 (x )))
        x =F .relu (self .bn4 (self .conv4 (x )))
        x =F .relu (self .bn5 (self .conv5 (x )))
        x =F .relu (self .bn6 (self .conv6 (x )))


        x =F .relu (self .bn7 (self .conv7 (x )))
        x =F .relu (self .bn8 (self .conv8 (x )))
        x =F .relu (self .bn9 (self .conv9 (x )))


        x =self .upsample1 (x )
        x =self .upsample2 (x )
        x =self .upsample3 (x )


        x =self .output (x )
        return x 


class SIGGRAPHColorizer (nn .Module ):
    """Implementation inspired by Zhang et al. SIGGRAPH 2017 paper: 
    Real-Time User-Guided Image Colorization with Learned Deep Priors"""
    def __init__ (self ):
        super (SIGGRAPHColorizer ,self ).__init__ ()


        self .low_level_features =nn .Sequential (
        nn .Conv2d (1 ,64 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .Conv2d (64 ,64 ,kernel_size =3 ,stride =2 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (64 ),
        nn .Conv2d (64 ,128 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .Conv2d (128 ,128 ,kernel_size =3 ,stride =2 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (128 ),
        nn .Conv2d (128 ,256 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .Conv2d (256 ,256 ,kernel_size =3 ,stride =2 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (256 ),
        nn .Conv2d (256 ,512 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .Conv2d (512 ,512 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (512 ),
        nn .Conv2d (512 ,256 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (256 ),
        )


        self .global_features =nn .Sequential (
        nn .Conv2d (256 ,512 ,kernel_size =3 ,stride =2 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (512 ),
        nn .Conv2d (512 ,512 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (512 ),
        nn .Conv2d (512 ,512 ,kernel_size =3 ,stride =2 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (512 ),
        nn .Conv2d (512 ,512 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (512 ),
        nn .AdaptiveAvgPool2d (1 ),
        nn .Conv2d (512 ,1024 ,kernel_size =1 ),
        nn .ReLU (True ),
        nn .Conv2d (1024 ,512 ,kernel_size =1 ),
        nn .ReLU (True ),
        )


        self .mid_level_features =nn .Sequential (
        nn .Conv2d (512 ,256 ,kernel_size =3 ,stride =1 ,padding =1 ),
        nn .ReLU (True ),
        nn .BatchNorm2d (256 ),
        )


        self .upsample1 =UpsampleBlock (256 ,128 )
        self .upsample2 =UpsampleBlock (128 ,64 )
        self .upsample3 =UpsampleBlock (64 ,32 )


        self .output =nn .Conv2d (32 ,2 ,kernel_size =1 ,stride =1 ,padding =0 )

    def forward (self ,x ):

        x =self .low_level_features (x )


        global_features =self .global_features (x )



        h ,w =x .size (2 ),x .size (3 )
        global_features =F .interpolate (global_features ,size =(h ,w ),mode ='bilinear',align_corners =False )


        fusion =x +global_features 


        fusion =self .mid_level_features (fusion )


        fusion =self .upsample1 (fusion )
        fusion =self .upsample2 (fusion )
        fusion =self .upsample3 (fusion )


        out =self .output (fusion )

        return out 


class ColorizationModelWithPerceptual (nn .Module ):
    def __init__ (self ,base_model =None ,vgg_features =None ):
        super (ColorizationModelWithPerceptual ,self ).__init__ ()


        if base_model is None :
            self .base_model =ECCVColorizer ()
        else :
            self .base_model =base_model 


        self .vgg_features =vgg_features 

    def forward (self ,x ,extract_features =False ,is_training =False ):

        ab_pred =self .base_model (x )

        if not extract_features or self .vgg_features is None :
            return ab_pred 



        return ab_pred 


def create_vgg_features ():
    """Create a VGG feature extractor for perceptual loss (optional)."""
    try :
        from torchvision .models import vgg16 
        from torchvision .models .feature_extraction import create_feature_extractor 


        vgg =vgg16 (pretrained =True )


        for param in vgg .parameters ():
            param .requires_grad =False 


        return create_feature_extractor (
        vgg ,
        return_nodes ={
        'features.4':'relu1_2',
        'features.9':'relu2_2',
        'features.16':'relu3_3'
        }
        )
    except Exception as e :
        print (f"VGG16 features not available: {e }")
        return None 


def create_model (use_perceptual =False ,use_gan =False ):
    """Create the colorization model and optional discriminator."""
    base_model =ECCVColorizer ()

    generator =None 
    if use_perceptual :
        vgg_features =create_vgg_features ()
        generator =ColorizationModelWithPerceptual (base_model ,vgg_features )
    else :
        generator =base_model 

    discriminator =None 
    if use_gan :

        discriminator =SIGGRAPHColorizer ()

    return (generator ,discriminator )if use_gan else generator 