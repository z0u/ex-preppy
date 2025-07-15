from .progress import Progress

__all__ = ['Progress']


# Example Usage (for testing in a notebook cell):
# from src.utils.progress import Progress
# import time
#
# total_items = 150
# items = range(total_items)
# metrics = {"loss": 1.5, "accuracy": 0.3}
#
# print("Iterating over Progress(range(total_items)):")
# with Progress(items, description="Training Epoch 1", initial_metrics=metrics) as pbar:
#     for i in pbar:
#         time.sleep(0.05)
#         if i % 10 == 0 and i > 0:
#             new_metrics = pbar.metrics.copy()
#             new_metrics["loss"] -= 0.05
#             new_metrics["accuracy"] += 0.02
#             pbar.update(0, metrics=new_metrics)
#         if i == 100:
#             pbar.update(0, suffix="Halfway there!")
#
# print("\nIterating over Progress(total=total_items):")
# with Progress(total=total_items, description="Processing Items") as pbar:
#     for i in pbar:
#         time.sleep(0.02)
#         if i == 50:
#             pbar.update(0, metrics={"status": "Phase 1 Complete"})
