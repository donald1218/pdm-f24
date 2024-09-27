# pdm-f24-hw1

NYCU Perception and Decision Making 2024 Fall

Spec: [Google Docs](https://docs.google.com/document/d/1QSbSWJ7s78h9QRS4EC3gsECFF8JDg0IT/edit?usp=sharing&ouid=101044242612677438105&rtpof=true&sd=true)

## Preparation
The replica dataset, you can use the same one in `hw0`.

## To execute hw1 program
Run ./run.sh.

    - The -l flag executes load.py and load.py -f 2.

    - The -c flag allows you to simultaneously reconstruct both the first and second floors.

    - The -b flag indicates that only bev.py will be executed, and reconstruct.py will not be executed.

    - The -r flag indicates that only reconstruct.py will be executed, and bev.py will not be executed.

If neither the -b nor -r flag is present, both bev.py and reconstruct.py will be executed.