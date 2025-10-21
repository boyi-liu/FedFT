import copy
import os

from transformers import TrainingArguments, Trainer
from alg.base import BaseClient, BaseServer
from datasets import load_dataset
from utils.model_utils import load_model


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        _, self.tokenizer = load_model(args)
        self.training_args = TrainingArguments(
            output_dir=f"./client{self.id}_lora-alpaca",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch"
        )
        self.load_data()

    def load_data(self):
        train_dir = os.path.join('./dataset', self.args.dataset, f'train/{self.id}.json')
        test_dir = os.path.join('./dataset', self.args.dataset, f'test/{self.id}.json')

        self.dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
        self.dataset['train'] = self.dataset['train'].map(self.format_example)
        self.dataset['test'] = self.dataset['test'].map(self.format_example)

    def format_example(self, example):
        prompt = f"Instruct: {example['question']}\nAnswer:"
        return {
            "input_ids": self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0],
            "labels": self.tokenizer(example["answer"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
        }

    def run(self, model):
        client_model = copy.deepcopy(model)

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()
        return {k: v for k, v in client_model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        client_model = copy.deepcopy(model)

        trainer = Trainer(
            model=client_model,
            args=self.training_args,
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer
        )
        metrics = trainer.evaluate()
        print(f"Client {self.id} local test metrics:", metrics)
        return metrics

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model, _ = load_model(args)
        self.client_models = []

        self.round = 0

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        pass

    def local_run(self):
        self.client_models = [client.run(self.model) for client in self.clients]

    def aggregate(self):
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)
        n = len(self.client_models)

        for model in self.client_models:
            for k, v in model.items():
                aggregated[k] = aggregated[k] + v

        aggregated_state_dict = {k: v / n for k, v in aggregated.items()}
        self.model.load_state_dict(aggregated_state_dict, strict=False)
        print("Aggregated model updated.")

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)

        if all_metrics and "eval_loss" in all_metrics[0]:
            avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
            print(f"\n[Server] 平均 eval_loss: {avg_loss:.4f}")