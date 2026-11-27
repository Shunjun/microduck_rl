"""All registered Microduck tasks share the project logging default."""

import unittest


class TrainingLoggerTests(unittest.TestCase):
    def test_default_covers_all_registered_tasks_and_allows_per_run_override(self):
        import mjlab.tasks  # Load the same task plugins as the training CLI.
        from mjlab.tasks.registry import list_tasks, load_rl_cfg

        tasks = [name for name in list_tasks() if 'MicroDuck' in name]
        self.assertGreater(len(tasks), 20)
        for task in tasks:
            with self.subTest(task=task):
                cfg = load_rl_cfg(task)
                self.assertEqual(cfg.logger, 'tensorboard')
                cfg.logger = 'wandb'  # CLI overrides a loaded config, not the registry.
                self.assertEqual(load_rl_cfg(task).logger, 'tensorboard')


if __name__ == '__main__':
    unittest.main()
