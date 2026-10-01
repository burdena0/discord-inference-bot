# third-party licensing review

reviewed on 2026-09-30 for publication of this source repository.

## scope and result

the repository imports dependencies through `requirements.txt`. it does not bundle their source, wheels, font files, model weights, or ollama binaries. no licensing blocker was identified for publishing the source files reviewed. this is a limited source review, not a legal clearance or a guarantee that the code is free of third-party material.

the original project code has no general reuse license. [COPYRIGHT](COPYRIGHT) applies only to rights held by the owner. it does not change dependency licenses.

## direct dependencies

| dependency | upstream license | source |
|---|---|---|
| discord.py | MIT | [license](https://github.com/Rapptz/discord.py/blob/master/LICENSE) |
| aiohttp | Apache-2.0 | [license](https://github.com/aio-libs/aiohttp/blob/master/LICENSE.txt) |
| python-dotenv | BSD-3-Clause | [license](https://github.com/theskumar/python-dotenv/blob/main/LICENSE) |
| numpy | BSD-3-Clause for core source | [license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pyfiglet | MIT for the python implementation; fonts have separate terms | [license and font discussion](https://github.com/pwaller/pyfiglet) |
| Pillow | MIT-CMU | [license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |

these licenses do not require the original bot source to be released under the same license merely because it imports these packages. when redistributing dependency code or binaries, preserve the applicable license texts, copyright notices, and any required notices. do not use upstream author names as endorsements.

requirements are not fully pinned. this review uses upstream license sources, not a resolved installation or a complete transitive-dependency audit. dependency versions, wheel contents, bundled native libraries, and transitive packages must be reviewed separately before publishing a packaged executable, container, or bundled environment. numpy and Pillow wheels can include additional third-party components with separate notices.

## font concern and mitigation

pyfiglet's upstream documentation distinguishes fonts with clear redistribution licenses from contributed fonts with less certain provenance. the bot previously exposed every installed font.

the public version limits font selection to `standard`, `small`, `mini`, `slant`, `big`, and `banner`. the headers for these files contain explicit BSD-style redistribution conditions. headers were reviewed at upstream commit `255827efbc09d662875dad50fa0edc1e1263bf45`:

- [standard](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/standard.flf)
- [small](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/small.flf)
- [mini](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/mini.flf)
- [slant](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/slant.flf)
- [big](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/big.flf)
- [banner](https://github.com/pwaller/pyfiglet/blob/255827efbc09d662875dad50fa0edc1e1263bf45/pyfiglet/fonts-standard/banner.flf)

the repo does not redistribute font files. a normal pyfiglet installation may still contain other fonts; the allowlist restricts bot use, not the contents of the installed package. do not bundle all installed fonts into a release without a separate review. installed font versions were not inspected in this source-only review.

## external models

model weights are downloaded separately and are not included here. the upstream [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and [nomic-embed-text-v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) identify Apache-2.0 licenses. exact ollama tags, downloaded artifacts, and alternate models must be checked separately. naming a model in configuration does not license its weights under this project's terms.

## platform terms

[GitHub's terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#5-license-grant-to-other-users) permit viewing and forking public repositories through the service. keeping this repo without an open-source license does not remove those platform permissions.
