#!/usr/bin/env python


def set_parameters():
    return {
        'batch_size': 32,
        'pool_size': 50,
        'base_lr': 0.0001,
        'max_epoch': 30,
        'lambda_cycle': 10, 
        'gamma_ide': 1,
        'domain_label': 5,
        'beta_gpwgan': 10,
        'condition': 'C5',
    }
