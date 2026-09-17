# KuaiRand-1K

Official documentation: https://kuairand.com/
Official archive record: https://zenodo.org/records/10439422
Domestic provider mirror: use the current "Chinese site" link published on
https://kuairand.com/ rather than copying a possibly stale deep link here.
File: KuaiRand-1K.tar.gz
Provider MD5: 6b0b9c8222d67fcd4c676218edca3f1f

Use 1K, not Pure: Pure removes out-of-pool interactions and has incomplete history.
Read provider terms, download the named archive and verify MD5 before extraction.
Expected layout is data/raw/KuaiRand-1K/data/; see prepare.json for the three CSVs.
Prepared manifests additionally record each used file's SHA-256.

Prefer the provider's domestic mirror. If it is unavailable, configure
`HTTPS_PROXY` and `HTTP_PROXY` in the download shell before accessing Zenodo.
Never store proxy credentials in this repository. Always verify the provider
MD5 before extracting, regardless of which endpoint supplied the archive.

Exposure rows use user_id, video_id, time_ms, is_click, is_rand and tab. The
content projection joins `video_features_basic_1k.csv` and maps video type,
upload type, tag, and upload date to OpenRec item content fields. User features
and `video_features_statistic_1k.csv` are intentionally excluded because their
historical availability has not been established. Ordinary/random exposure
strata remain separate. Inspect tab semantics before interpreting is_click.
