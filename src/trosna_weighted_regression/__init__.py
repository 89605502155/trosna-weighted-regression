"""Взвешенная по времени линейная регрессия с защитой от засыпания при резких изменениях."""

from trosna_weighted_regression.core import KERNELS, WakefulWLS, kernel_weights

__all__ = ["KERNELS", "WakefulWLS", "kernel_weights"]
