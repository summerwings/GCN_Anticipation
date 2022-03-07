from infer_GCN_head import infer_GCN
import logging

logging.basicConfig(level=logging.DEBUG,
                    filename='infer.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

Infer = infer_GCN()
Infer.infer()