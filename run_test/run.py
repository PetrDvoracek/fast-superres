import os
import argparse
import json
from src.generator import ImageGenerator
from src.runner import Runner
from src.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-dir', default = './submit')
    parser.add_argument('--input-data-dir', default = './validation')
    parser.add_argument('--input-params-path', default = './input.json')
    parser.add_argument('--result-dir', default = './results')
    parser.add_argument('--result-name', default = 'scores.json')

    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()

    # set the input data and instanciate the generator
    with open(args.input_params_path) as f:
        input_data = json.load(f)
    sample_data = input_data['samples']
    params = {k: os.path.abspath(os.path.join(args.input_data_dir, v)) for k, v in input_data['params'].items()}
    generator = ImageGenerator(sample_data=sample_data, params=params)
    result_dir = os.path.abspath(args.result_dir)
    exec_dir = os.path.abspath(args.exec_dir)
    os.makedirs(result_dir, exist_ok=True)

    # set the predictor and load the model or other pre-set data
    Predictor.get_model(model_path=exec_dir)

    # run the inference
    runner = Runner(predictor=Predictor, generator=generator)
    runner.run()
    result, runtime = runner.get_result()

    # preprocess and save the result
    print('Mean PSNR: {}'.format(sum(result.values())/len(result)))
    print('Inference Speed: {} [s/frame]'.format(sum(runtime.values())/len(runtime)))
    result_all = {}
    for k in result.keys():
        result_all[k] = {'psnr': result[k], 'runtime': runtime[k]}
    with open(os.path.join(result_dir, args.result_name), 'w') as f:
        json.dump(result_all, f, indent=4)
    

if __name__ == '__main__':
    main()
