
## Neural-Pull: 从点云中学习有符号距离函数，通过学习将空间拉到曲面上 (ICML 2021)


这个代码库包含了论文 [Neural-Pull: 从点云中学习有符号距离函数，通过学习将空间拉到曲面上](https://arxiv.org/abs/2011.13495) 的官方 PyTorch 版本代码。

您可以在下面找到详细的使用说明，用于训练您自己的模型或运行我们提供的演示。

如果您觉得我们的代码或论文有用，请考虑引用：

    @inproceedings{NeuralPull,
        title = {Neural-Pull: 从点云中学习有符号距离函数，通过学习将空间拉到曲面上},
        author = {马宝瑞 and 韩志中 and 刘宇深 and Matthias Zwicker},
        booktitle = {国际机器学习大会 (ICML)},
        year = {2021}
    }

## Tensorflow 版本
这个代码库包含了 Neural-Pull 的官方 PyTorch 版本代码。如果您更喜欢使用 TensorFlow 代码，请使用 [tensorflow 代码库并给它点赞](https://github.com/mabaorui/NeuralPull)，谢谢。

## 曲面重建演示
<p align="left">
  <img src="img/input_poitns.png" width="410" /><img src="img/Demo.jpg" width="370" />
</p>

## 安装:
我们的代码是在 Python 3.8、PyTorch 1.11.0 和 CUDA 11.3 上实现的。
- 安装 Python 依赖
```bash
conda create -n npull python=3.8
conda activate npull
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install tqdm pyhocon==0.3.57 trimesh PyMCubes scipy
```

## 数据:
从点云进行曲面重建。

我们在 `data/gargoyle.ply` 中提供了演示数据。如果您想重建自己的数据，请：
- 将您的点云数据放在 `./data` 目录下。
- 请注意，我们支持 `.ply` 和 `.xyz` 格式的点云数据。

## 使用:
```python
python run.py --gpu 0 --conf confs/npull.conf --mode validate_mesh --dataname maskLoss --dir maskLoss
python run.py --gpu 0 --conf confs/npull.conf --mode validate_mesh --dataname gargoyle --dir gargoyle
python run.py --gpu 0 --conf confs/npull.conf --dataname YOUR_DATANAME --dir YOUR_OUTPUT_DIR
```
```
您可以在 `./outs` 目录中找到生成的网格和日志。

## 许可证
本项目在 MIT 许可证下开源。
