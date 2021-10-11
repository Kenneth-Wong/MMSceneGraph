import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader


class EvalHook(Hook):
    """Evaluation hook.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, relation_mode=False, relcaption_mode=False,
                 downstream_caption_mode=False, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.relation_mode = relation_mode
        self.relcaption_mode = relcaption_mode
        self.downstream_caption_mode = downstream_caption_mode
        self.eval_kwargs = eval_kwargs
        self.key_first = self.eval_kwargs.pop('key_first', False)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, self.relation_mode, self.relcaption_mode,
                                  self.downstream_caption_mode, show=False,
                                  key_first=self.key_first)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        #For some image caption datasets, it may need to save some results during evaluation.
        if self.dataloader.dataset.__class__.__name__ in ['GeneralizedVisualGenomeDataset', 'CaptionCocoDataset']:
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, epoch=runner.epoch, work_dir=runner.work_dir, **self.eval_kwargs)
        else:
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(Hook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 relation_mode=False,
                 relcaption_mode=False,
                 downstream_caption_mode=False,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.relation_mode = relation_mode
        self.relcaption_mode = relcaption_mode
        self.downstream_caption_mode = downstream_caption_mode
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.key_first = self.eval_kwargs.pop('key_first', False)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            relation_mode=self.relation_mode,
            relcaption_mode=self.relcaption_mode,
            downstream_caption_mode=self.downstream_caption_mode,
            key_first=self.key_first,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def evaluate(self, runner, results):
        # For some image caption datasets, it may need to save some results during evaluation.
        if self.dataloader.dataset.__class__.__name__ in ['GeneralizedVisualGenomeDataset', 'CaptionCocoDataset']:
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, epoch=runner.epoch, work_dir=runner.work_dir, **self.eval_kwargs)
        else:
            eval_res = self.dataloader.dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
