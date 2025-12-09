package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	// Create a shared context
	ctx := context.Background()

	// Run the stages of the pipeline
	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// python := client.Container().From("python:3.12.2-bookworm").
	// 	WithDirectory("python", client.Host().Directory("../Module")).
	// 	WithExec([]string{"python", "--version"})

	python := client.Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/src", client.Host().Directory("../")).               // mounts entire repo
		WithWorkdir("/src/Module").                                          // set workdir to Module
		WithExec([]string{"pip", "install", "-r", "/src/requirements.txt"}). // repo root
		WithExec([]string{"python", "0.1_read_data.py"})                     // workdir script

	_, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

	return nil
}
