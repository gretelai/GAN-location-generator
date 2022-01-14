from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        return [
            # Command 0
            Options(
                dataroot="./datasets/ebike_locations",
                name="locations_CUT",
                CUT_mode="CUT",
            ),
            # Command 1
            Options(
                dataroot="./datasets/ebike_locations",
                name="locations_FastCUT",
                CUT_mode="FastCUT",
            )
        ]

    def commands(self):
        return ["python train.py " + str(opt.set(n_epochs=50)) for opt in self.common_options()]

    def test_commands(self):
        return ["python test.py " + str(opt.set(num_test=500, phase='test', preprocess='scale_width', load_size=256))
                for opt in self.common_options()]
