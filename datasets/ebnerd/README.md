# EB-NeRD

Official data and license: https://recsys.eb.dk/
Official benchmark: https://github.com/jppol-ai/ebnerd-benchmark

Obtain the small version through the official download flow after reviewing its
terms. Place train/behaviors.parquet, validation/behaviors.parquet and
articles.parquet under data/raw/ebnerd_small/ (or edit prepare.json).
Keep history.parquet for the planned history adapter; v1 does not use it.

Only labeled train/validation are supported locally. Hidden official test needs
an inference-only submission workflow, which is not implemented yet.
No archive URL is hardcoded to bypass the provider's download terms. Do not
commit or redistribute raw records with the Apache-licensed experiment code.

If the official site must be reached through a domestic proxy, configure it in
the shell that performs the authorized download, for example:

```bash
export HTTPS_PROXY=http://127.0.0.1:<port>
export HTTP_PROXY="$HTTPS_PROXY"
```

Do not commit proxy credentials. Unset both variables after the download.
