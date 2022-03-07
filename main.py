from main_GCN_head import  train_GCN
import logging


logging.basicConfig(level=logging.DEBUG,
                    filename='train.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

Train = train_GCN(1032022)
Train.train()
