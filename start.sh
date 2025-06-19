#!/bin/bash
until python main.py tangle-fl targeted1; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl targeted25; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted1; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted25; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl untargeted1; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl untargeted25; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted1; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted25; do
    echo "Command failed, retrying..."
done
