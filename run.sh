torch-model-archiver --model-name "gpt2" --version 1.0 --serialized-file ./wodehouse_model/pytorch_model.bin --extra-files "./wodehouse_model/config.json,./wodehouse_model/vocab.json,./wodehouse_model/merges.txt" --handler "./wodehouse_generator_ts_handler.py"

## move gpt2.mar file to wodehouse_model_store
mv gpt2.mar wodehouse_model_store/

torchserve --start --model-store wodehouse_model_store --models gpt2=gpt2.mar --no-config-snapshots

## torchserve --stop

## curl -X POST http://127.0.0.1:8080/predictions/gpt2 -T prompt.txt

## curl -X GET http://127.0.0.1:8081/models
