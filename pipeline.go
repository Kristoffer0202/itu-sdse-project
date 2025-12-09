package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	// Optional CLI experiment argument
	experiment := "default_exp"
	if len(os.Args) > 1 {
		experiment = os.Args[1]
	}

	if err := Build(ctx, experiment); err != nil {
		fmt.Println("Error:", err)
		panic(err)
	}
}

func Build(ctx context.Context, experiment string) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().
		From("python:3.12.2-bookworm").
		WithDirectory("/src", client.Host().Directory(".")).
		WithWorkdir("/src/Module").
		// Install dependencies
		WithExec([]string{"pip", "install", "-r", "/src/requirements.txt"}).
		WithExec([]string{"pip", "install", "dvc"})

	// Scripts in order
	scripts := []string{
		"0.1_read_data.py",
		"0.2_data_preprocessing.py",
		"1.0_Train.py",
		"1.1_TrainXGBoost.py",
		"1.2_TrainSKLearnLog.py",
		"2.0_SelectBestModelAndRegister.py",
		"3.0_Deploy.py",
	}

	// Scripts that need timestamp experiment name
	trainingScripts := map[string]bool{
		"1.0_Train.py":                      true,
		"1.1_TrainXGBoost.py":               true,
		"1.2_TrainSKLearnLog.py":            true,
		"2.0_SelectBestModelAndRegister.py": true,
	}

	// Generate timestamp once
	timestampExp := time.Now().Format("20060102_150405")

	for _, script := range scripts {
		arg := experiment
		if trainingScripts[script] {
			arg = timestampExp
		}
		// Use --run_name only for training scripts
		if trainingScripts[script] {
			python = python.WithExec([]string{"python", script, "--run_name", arg})
		} else {
			python = python.WithExec([]string{"python", script, arg})
		}
	}
	// Export output if needed
	_, err = python.Directory("artifacts").Export(ctx, "artifacts")
	if err != nil {
		return err
	}

	return nil
}
