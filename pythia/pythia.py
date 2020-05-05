import argparse
import time

from pythia import io
from pythia import svm

import logging
logging.basicConfig(
    level=logging.NOTSET,
    format='%(message)s'
)
logger = logging.getLogger('pythia')

def fit(args):
    image_folder = args.image_folder
    data_filename = args.data
    clf_output = args.clf_output
    
    logger.info('Fitting data ...')
    svc = SVC(
        data_filename,
        image_folder
    )
    logger.info('Fitting data complete!')
    logger.info('Saving classifier ...')
    io.clfsave(
        clf_output,
        svc
    )
    logger.info('Saving classifier complete!')
    

def predict(args):
    clf_input = args.clf_input
    image_filename = args.image
    
    logger.info('Loading classifier ...')
    svc = io.clfread(
        clf_input
    )
    logger.info('Loading classifier complete!')
    logger.info('Loading image ...')
    image = io.imread(
        image_filename
    )
    logger.info('Loading image complete!')
    logger.info('Predicting image ...')
    prediction = svc.predict(
        image
    )
    logger.info('Predicting image complete')
    print("Image is {}".format(prediction))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CRIC SVM.'
    )
    subparsers = parser.add_subparsers()
    
    parser.add_argument(
        '--clock',
        action='store_true',
        help='Clock time of operation'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose mode'
    )
    
    # Parser for training
    parser_fit = subparsers.add_parser('fit')
    parser_fit.add_argument(
        '--image-folder',
        default="img",
        help='Image folder for training'
    )
    parser_fit.add_argument(
        '--data',
        default="classifications.json",
        help='Data for training in JSON format'
    )
    parser_fit.add_argument(
        '--clf-output',
        default="cric-svm.joblist",
        help='Output file with the SVM classifier'
    )
    parser_fit.set_defaults(
        func=fit
    )
    
    # Parser for predict
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument(
        '--clf-input',
        default="cric-svm.joblist",
        help='Input file with the SVM classifier'
    )
    parser_predict.add_argument(
        '--image',
        default="sample.png",
        help='Input image file to be classifier'
    )
    parser_predict.set_defaults(
        func=predict
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
        
    if args.clock:
        t0 = time.time()
    
    args.func(args)
    
    if args.clock:
        print(
            """{} seconds wall time""".format(
                time.time() - t0
            )
        )