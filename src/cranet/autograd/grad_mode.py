from .tensor import is_grad_enabled, set_grad_enabled


class no_grad:
    def __init__(self):
        self.prev = False

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)
