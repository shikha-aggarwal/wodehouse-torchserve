Use pytorch/serve [https://github.com/pytorch/serve] to serve a GPT-2 model.
The model was trained using the Hugging Face library on a collection of Wodehouse novels.
More details in this colab file: https://github.com/shikha-aggarwal/wodehouse-generator/blob/main/gpt2_huggingface.ipynb

I have not included the model files in the repo because they are too large. I would recommend running the above colab to build the model.
You can then put the model in a subdirectory (wodehouse_model) in this folder.

How to run
----------

1. git clone git@github.com:shikha-aggarwal/wodehouse-torchserve.git
2. Create a venv environment and activate
3. git clone https://github.com/pytorch/serve.git
4. python ./ts_scripts/install_dependencies.py
5. pip install torchserve torch-model-archiver
6. pip install requirements.txt
7. ./run.sh
