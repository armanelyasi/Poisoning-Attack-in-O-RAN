# *How to Poison an xApp*: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks

*Andrea Lacava*$^a$*, Stefano Maxenti*$^a$*, Leonardo Bonati*$^a$*, Salvatore D’Oro*$^a$*, Alina Oprea*$^b$*, Tommaso Melodia*$^a$*, Francesco Restuccia*$^a$

$^a$ Institute for the Wireless Internet of Things, Northeastern University, Boston, MA, USA

$^b$ Khoury College of Computer Sciences, Northeastern University, Boston, MA, USA

The development of Open Radio Access Network (RAN) cellular systems is being propelled by the integration of Artificial Intelligence (AI) techniques. While AI can enhance network performance, it expands the attack surface of the RAN. For instance, the need for datasets to train AI algorithms and the use of open interface to retrieve data in real time paves the way to data tampering during both training and inference phases. In this work, we propose MalO-RAN, a framework to evaluate the impact of data poisoning on O-RAN intelligent applications. We focus on AI-based xApps taking control decisions via Deep Reinforcement Learning (DRL), and investigate backdoor attacks, where tampered data is added to training datasets to include a backdoor in the final model that can be used by the attacker to trigger potentially harmful or inefficient pre-defined control decisions. We leverage an extensive O-RAN dataset collected on the Colosseum network emulator and show how an attacker may tamper with the training of AI models embedded in xApps, with the goal of favoring specific tenants after the application deployment on the network. We experimentally evaluate the impact of the SleeperNets and TrojDRL attacks and show that backdoor attacks achieve up to a 0.9 attack success rate. Moreover, we demonstrate the impact of these attacks on a live O-RAN deployment implemented on Colosseum, where we instantiate the xApps poisoned with MalO-RAN on an O-RAN-compliant Near-real-time RAN Intelligent Controller (RIC). Results show that these attacks cause an average network performance degradation of 87%.

If you use the Mal-O-RAN concept and/or the framework in your research, please cite the following paper:

> A. Lacava, S. Maxenti, L. Bonati, S. D’Oro, A. Oprea, T. Melodia, and F. Restuccia, “How to Poison an xApp: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks,” Computer Networks, vol. 273, p. 111727, 2025 [[pdf]](https://www.sciencedirect.com/science/article/pii/S1389128625006930) [[doi]](https://doi.org/10.1016/j.comnet.2025.111727) [[bibtex]](docs/cite.bib)

The official bibtex is reported here:

```text
@article{LACAVA2025111727,
	title        = {{How to Poison an xApp: Dissecting Backdoor Attacks to Deep Reinforcement Learning in Open Radio Access Networks}},
	author       = {Andrea Lacava and Stefano Maxenti and Leonardo Bonati and Salvatore D’Oro and Alina Oprea and Tommaso Melodia and Francesco Restuccia},
	year         = 2025,
	journal      = {Computer Networks},
	volume       = 273,
	pages        = 111727,
	doi          = {https://doi.org/10.1016/j.comnet.2025.111727},
	issn         = {1389-1286},
	url          = {https://www.sciencedirect.com/science/article/pii/S1389128625006930},
	keywords     = {Open RAN, 5G, AI, Adversarial AI, DRL},
	abstract     = {The development of Open Radio Access Network (RAN) cellular systems is being propelled by the integration of Artificial Intelligence (AI) techniques. While AI can enhance network performance, it expands the attack surface of the RAN. For instance, the need for datasets to train AI algorithms and the use of open interface to retrieve data in real time paves the way to data tampering during both training and inference phases. In this work, we propose MalO-RAN, a framework to evaluate the impact of data poisoning on O-RAN intelligent applications. We focus on AI-based xApps taking control decisions via Deep Reinforcement Learning (DRL), and investigate backdoor attacks, where tampered data is added to training datasets to include a backdoor in the final model that can be used by the attacker to trigger potentially harmful or inefficient pre-defined control decisions. We leverage an extensive O-RAN dataset collected on the Colosseum network emulator and show how an attacker may tamper with the training of AI models embedded in xApps, with the goal of favoring specific tenants after the application deployment on the network. We experimentally evaluate the impact of the SleeperNets and TrojDRL attacks and show that backdoor attacks achieve up to a 0.9 attack success rate. Moreover, we demonstrate the impact of these attacks on a live O-RAN deployment implemented on Colosseum, where we instantiate the xApps poisoned with MalO-RAN on an O-RAN-compliant Near-real-time RAN Intelligent Controller (RIC). Results show that these attacks cause an average network performance degradation of 87%.}
}
```

![Mal-O-RAN](./docs/images/maloran.png "Mal-O-RAN")

MalO-RAN is a modular and extensible framework for understanding and mitigating security risks related to the Adversarial AI in the Open RAN.

It is composed of three different entities:

- the RAN environment which is an offline environment (This repository).
- The RIC and the xApps that are used in the evaluation phase (Not available at the moment).
- Adversarial pipeline that is able to train an agent and craft data poisoning attacks (This repository).

## Usage

Before starting, download your dataset. In this work, we used the Col-O-RAN datasetand we cloned [this repository](https://github.com/wineslab/colosseum-oran-coloran-dataset). Insert the parent folder of the dataset in the [build script](./docker/build.sh).

1) Run `export PYTHONPATH=/workspace/mal-o-ran/src` on your terminal if there are path issues

2) Insert your `WANDB_API_KEY` in `.src/utils/constants.py`

3) Create the dataset and perform the preprocessing:

    ```bash
    python3 src/etl/extractor.py
    ```

4) Training example (for a more complex one refer to `start_batch.sh`). Results are stored in wandb

    ```bash
    python3 src/sleeper_nets/ppo.py --num_envs 1 --target_action 0 --strong --trojdrl --total_timesteps 1500000  --p_rate .5 --rew_p 0.1
    ```

5) Results:
    Check wandb and eventually download the csvs. Some useful scripts for the analysis and the plotting can be found in [here](./src/utils/analysis.py).

## Acknowledgment

This work has been supported  in part by the National Science Foundation under grants CNS-2312875 and OAC-2530896; by the Air Force Office of Scientific Research under grant FA9550-23-1-0261; by the Office of Naval Research under grant N00014-23-1-2221; by OUSD(R\&E) through Army Research Laboratory Cooperative Agreement Number W911NF-24-2-0065. The work was also partially supported by SERICS (PE00000014) 5GSec project, CUP B53C22003990006, under the MUR National Recovery and Resilience Plan funded by the European Union - NextGenerationEU, and by the U.S. National Science Foundation under grants CNS-1925601, CNS-2312875, and CNS-2112471. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

The authors would also like to acknowledge Dr. Giorgio Severi and Ethan Rathbun for providing resources on poisoning attacks and for their participation in discussions related to the project.
