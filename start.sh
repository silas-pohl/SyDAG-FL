#!/bin/bash
until python main.py tangle-fl untargeted1; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl untargeted5; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl untargeted25; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl untargeted50; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted1; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted5; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted25; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl untargeted50; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl targeted1; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl targeted5; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl targeted25; do
    echo "Command failed, retrying..."
done
until python main.py tangle-fl targeted50; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted1; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted5; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted25; do
    echo "Command failed, retrying..."
done
until python main.py sydag-fl targeted50; do
    echo "Command failed, retrying..."
done
