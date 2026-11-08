# LoopLLM Kaggle + Google Drive setup

## Kaggle input layout

Kaggle may expose the Dataset like this:

```text
/kaggle/input/datasets/<user>/loop-llm/
├── train.bin
├── train.val.bin
└── loopllm_moa7/
    └── loopllm_moa7/
        └── loopllm_clean_rfc/
            └── train.py
```

The trainer discovers `train.bin` and `train.val.bin` automatically when
`--source Kaggle` is used. The source code can remain under `/kaggle/input`;
it does not need to be copied just to run it.

## Google Drive credential

The Drive backend uses PyDrive2 plus a Kaggle Secret. Kaggle's `UserSecretsClient`
reads the secret at runtime, so credentials do not have to be embedded in the
notebook/source. See Kaggle's current Secrets documentation and PyDrive2's
OAuth documentation.

1. Create a Google OAuth client for a desktop/local application in Google Cloud.
2. Download the OAuth client JSON as `client_secrets.json`.
3. On a local machine with PyDrive2 installed, run:

   ```bash
   python scripts/create_drive_credentials.py
   ```

4. Copy the complete generated `credentials.json` contents into a Kaggle Secret
   with the exact label:

   ```text
   GDRIVE_CREDENTIALS_JSON
   ```

5. Attach that Secret to the Kaggle Notebook under **Add-ons -> Secrets**.

Kaggle documents `UserSecretsClient().get_secret(...)` for runtime secret access.
PyDrive2 documents `GoogleAuth.LoadCredentialsFile()` / refreshable credentials
and its OAuth flows.

## Install Drive dependency in Kaggle

Your existing model environment should keep Kaggle's supplied PyTorch build.
Only install the Drive dependency:

```python
%pip install -q PyDrive2
```

Restart the session once if Kaggle reports that an updated package needs to be
loaded.

## Run

Your normal training command stays unchanged except for:

```text
--source Kaggle
```

In Kaggle mode the trainer maps the old Colab paths automatically:

```text
/content/drive/MyDrive/train.bin
    -> /kaggle/input/.../train.bin

/content/drive/MyDrive/train.val.bin
    -> /kaggle/input/.../train.val.bin

/content/drive/MyDrive/loop_llm/...
    -> /kaggle/working/loop_llm/...
```

At every Kaggle launch, the complete remote Google Drive `loop_llm` tree is
restored to `/kaggle/working/loop_llm` before path validation and training. With
`torchrun`, only rank 0 performs the restore and then broadcasts success or failure
to the other rank. Startup logs show authentication, folder listing, file download
progress, and completion so a slow or blocked restore is distinguishable from a
CUDA/NCCL startup issue. The current training entrypoint does not start an automatic
Drive-upload worker; local training outputs must still be uploaded separately.

For both T4 GPUs, launch the command with `torchrun --standalone
--nproc-per-node=2 train.py ...`. Use `python train.py ...` for one GPU.
