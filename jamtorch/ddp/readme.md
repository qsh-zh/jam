## Warning

* Adjust learning rate by user
* Log only show the master performance (additional evaluation code is needed)
* Resume on optimizer is difficult


- [ ] be careful about amp fp16!
- [ ] create a fake ddp_trainer, same api

### Syn ddp and non-ddp

* setup ddp
* `mp.spawn` vs directly call methods
