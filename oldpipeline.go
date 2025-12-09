package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	if err := Build(ctx); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/src", client.Host().Directory(".")).
		WithWorkdir("/src/Module").
		WithExec([]string{"pip", "install", "-r", "/src/requirements.txt"}).
		WithExec([]string{"pip", "install", "dvc"}).
		WithExec([]string{"python", "0.1_read_data.py"})

	// Export the correct directories
	_, err = python.Directory("/src/data").Export(ctx, "output/data")
	if err != nil {
		return err
	}

	_, err = python.Directory("/src/notebooks/artifacts").Export(ctx, "output/artifacts")
	if err != nil {
		return err
	}

	return nil
}
