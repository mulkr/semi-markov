# Modeling hidden semi-Markov models

> [!CAUTION]
> This project is currently under development.
> Expect things to break and change

## Inspirations

* https://github.com/jvkersch/hsmmlearn<br>
I am familiar with its structure. Internal implementation in c++, with a style that I personally don't like. Documentation is sparse, but it is there. Archived. 
* https://github.com/poypoyan/edhsmm<br>
Another implementation. No proper documentation. Archived.
* https://github.com/GarroshIcecream/ChadHMM<br>
Most up to date. But uses pyTorch, which is overkill in my opinion.
* https://github.com/hmmlearn/hmmlearn<br>
Not semi-Markov but good reference.

> [!IMPORTANT]
> I have reused some code from these projects.
> The reused code was licensed under GPLv3 but I have not yet labeled the changes made, this will be fixed in the future

## Installation
### From source
```console
git clone <repo>
cd <repo>
pip install .
```
Or use any `pip install` equivalent. You should also consider installing in a virtual environment.

> [!NOTE]
> Since the Viterbi and forward-backward fitting algorithms will eventually be written in C, in the future you will must have a C compiler set up.
> This can be gcc (MinGW on windows), clang, MSVC or any other compiler supporting the C23 standard and able to compile to `.so` or `.dll` files.

## Correspondence
Kristóf Müller (muller.kristof@itk.ppke.hu)