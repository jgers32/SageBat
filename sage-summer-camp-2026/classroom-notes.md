# Classroom Notes

- [ECR documentation](https://sagecontinuum.org/docs/tutorials/edge-apps/intro-to-edge-apps)
- to update app, there is a `New` button where it pulls the github repo (commit changes first) then just update the version number

- camera app [repo](https://github.com/jgers32/julia-hack-app)


## List 
- [x] created, published app, and ran camera app on H034
- [x] set-up Sage MCP

---

matthew thompson (research SWE @ OSU imageomics)

[https://imageomics.github.io/catalog/#type=models](https://imageomics.github.io/catalog/#type=models)

- domain specialization/adaption:
    - continued pretraining
    - fine-tuning: subset of weights (full/param-efficient)
    - few-shot probing

- deployment optimization:
    - distillation
    - quantization

- edge constraints guiding design:
    - hardware limits
    - science requirements
    - pratical limitations

PTQ: post-training quant
QAT: q-aware training
