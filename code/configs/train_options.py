from configs.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs',      type=int,   default=80)
        parser.add_argument('--lr',          type=float, default=1e-4)
        
        parser.add_argument('--crop_h',  type=int, default=448)
        parser.add_argument('--crop_w',  type=int, default=576)        
        parser.add_argument('--log_dir', type=str, default='./logs')

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--save_model', action='store_true')        
        parser.add_argument('--save_result', action='store_true')
        
        parser.add_argument('--warm_up_end',            type=int,   default=100)
        parser.add_argument('--learning_rate_alpha',    type=float,   default=0.05)
sse+                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        return parser

# test-eval
#   d1,     d2,     d3,     AbsRel, SqRel,  RMSE,   RMSElog,    SILog,  log10
#   0.967,  0.996,  0.999,  0.057,  0.187,  2.297,  0.086,      7.923,  0.025

# train after 20 epochs
#   d1      d2      d3      abs_rel sq_rel  rmse    rmse_log    silog   log10
# 0.8693    0.9737  0.9955  0.1056  0.5007  3.4351  0.1549      0.1445  0.0437

# after 20 epochs-update_lr
#     d1    d2      d3      abs_rel sq_rel  rmse    rmse_log    log10   silog 
# 0.8735    0.9697  0.9911  0.1043  0.4804  3.4024  0.136      0.0464  0.1441 5

# after 80 epochs
#     d1    d2      d3      abs_rel sq_rel  rmse    rmse_log    log10   silog 
# 0.8918    0.9788  0.9941  0.0921  0.4170  3.2493  0.1414      0.0402  0.1338 