#!/bin/sh
# Simple health check script for Ollama
if nc -z localhost 11434; then
    exit 0
else
    exit 1
fi
