# SageBat

1. Adding on 384kHz ultrasound microphone to next-gen Sage node; to be used for acoustic detection, species identification, and isolating bat chirps from background noise (chirps = ms micro events, start with band-pass filtering & energy-based onset detection). Could also be a trigger to the mobotix (thermal) modality on the same node. 
2. Transmitting ultrasound data (especially long duration clips..) require lots of compute and network resources -- so instead, learn the sparse representation and the embedding size (using the structure of the data) on the edge to transmit. 
