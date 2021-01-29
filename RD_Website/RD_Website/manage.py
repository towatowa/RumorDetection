#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from RD.rumor_model.liumeiqi.rumor_detection import MyGRU
from RD.rumor_model.chenfuguan.myUtils import BiRNN
from RD.rumor_model.my_edr import ERD_CN, dataUtils_CN, RDM_Model, CM_Model, config


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'RD_Website.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
