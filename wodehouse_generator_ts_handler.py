from abc import ABC
import json
import logging
import os

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes. 
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=False,
            return_tensors="pt"
        )
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        # NOTE: This makes the assumption that your model expects text to be tokenized  
        # with "input_ids" and "token_type_ids" - which is true for some popular transformer models, e.g. bert.
        # If your transformer model expects different tokenization, adapt this code to suit 
        # its expected input format.
        # prediction = self.model(
        #     inputs['input_ids'].to(self.device),
        #     token_type_ids=inputs['token_type_ids'].to(self.device)
        # )[0].argmax().item()
        logger.info("inputs: '%s'", inputs)

        prediction = self.model.to(self.device).generate(
            input_ids=inputs['input_ids'].to(self.device),
            do_sample=True,
            max_length=200,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            repetition_penalty=None,
            num_return_sequences=3,
        ) #[0].argmax().item()

        logger.info("Model predicted shape: '%s'", prediction.shape)

        return prediction

    def postprocess(self, inference_output):

        logger.info("inference_output: %s", inference_output)

        if len(inference_output.shape) > 2:
                inference_output.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(inference_output):
            print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            generated_sequences.append(text)
            #print(text)

        return [generated_sequences]


        # return inference_output


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
