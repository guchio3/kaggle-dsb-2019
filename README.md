# kaggle-dsb-2019
### NOTES
 - every codes should be run at the repo root dir


### env settings and its basic usage
 1. build kaggle gpu image in your local env (because it's based on kaggle gpu image, which does not exist some-hub officially)
     - `git clone git@github.com:Kaggle/docker-python.git; cd docker-python; ./build --gpu` 
 1. clone this repo
     - `cd; git clone git@github.com:guchio3/kaggle-google-quest.git; cd kaggle-google-quest`
 1. run commands
     - shell           : `docker-compose run shell`
         - ex. debug using pudb
     - python commands : `docker-compose run python {something.py}`
         - train : ``
         - predict : ``
     - notebooks       : `docker-compose run --service-ports jn`
