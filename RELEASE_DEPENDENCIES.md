# Release dependency and license inventory

Generated 2026-10-21 using `D:\Users\Hp\Documents\ChatGPT\3d\.venv-gpu\Scripts\python.exe` distribution metadata.
Python: `3.12.10` (Windows-11-10.0.22000-SP0).

This combines installed-package metadata from the interpreter running this script with a read-only inspection of the selected PyInstaller archive. It is not a lock file, a complete frozen SBOM, or a legal opinion. Verify full upstream license texts, notices, native component provenance, and delivery materials before redistribution.

## Project source license

The repository owner directed broad public reuse. The repository's original
StereoLab source and documentation are now licensed under MIT; see the root
`LICENSE` file. This project license does not change the separate terms for
third-party packages, native runtime files, model weights/conversions, or
FFmpeg. It does not by itself establish that a particular executable or model
can be redistributed. The candidate-specific review and notice gaps below still
apply.

## Direct runtime requirements

The version ranges come from `requirements-gpu.txt`; resolved versions below describe the interpreter running this script, not necessarily every file frozen into the selected EXE.

| Requirement | Resolved version |
| --- | --- |
| `fastapi` | `0.143.3` |
| `starlette` | `1.8.0` |
| `uvicorn` | `0.54.0` |
| `python-multipart` | `0.0.32` |
| `numpy` | `2.5.4` |
| `opencv-python` | `5.0.0.93` |
| `onnxruntime-directml` | `1.24.4` |
| `onnx` | `1.23.2` |

## Runtime dependency closure

| Distribution | Version | License metadata |
| --- | --- | --- |
| `annotated-doc` | `0.0.5` | MIT |
| `annotated-types` | `0.8.0` | MIT |
| `anyio` | `4.15.1` | MIT |
| `click` | `8.5.0` | BSD-3-Clause |
| `fastapi` | `0.143.3` | MIT |
| `flatbuffers` | `25.12.19` | Apache Software License |
| `h11` | `0.16.0` | MIT License |
| `idna` | `3.20` | BSD-3-Clause |
| `ml_dtypes` | `0.6.0` | Apache-2.0 |
| `mpmath` | `1.3.0` | BSD License |
| `numpy` | `2.5.4` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `onnx` | `1.23.2` | Apache-2.0 |
| `onnxruntime-directml` | `1.24.4` | MIT License |
| `opencv-python` | `5.0.0.93` | Apache Software License |
| `opentelemetry-api` | `1.45.1` | Apache-2.0 |
| `packaging` | `26.3` | Apache-2.0 OR BSD-2-Clause |
| `protobuf` | `7.36.2` | 3-Clause BSD License |
| `pydantic` | `2.14.1` | MIT |
| `pydantic_core` | `2.50.1` | MIT |
| `python-multipart` | `0.0.32` | Apache-2.0 |
| `starlette` | `1.8.0` | BSD-3-Clause |
| `sympy` | `1.14.0` | BSD License |
| `typing_extensions` | `4.16.0` | PSF-2.0 |
| `typing-inspection` | `0.4.4` | MIT |
| `uvicorn` | `0.54.0` | BSD-3-Clause |

## Build-time dependency closure

| Distribution | Version | License metadata |
| --- | --- | --- |
| `altgraph` | `0.17.5` | MIT License |
| `packaging` | `26.3` | Apache-2.0 OR BSD-2-Clause |
| `pefile` | `2024.8.26` | MIT |
| `pyinstaller` | `6.22.3` | GNU General Public License v2 (GPLv2) |
| `pyinstaller-hooks-contrib` | `2026.8` | Apache Software License; GNU General Public License v2 (GPLv2) |
| `pywin32-ctypes` | `0.2.3` | BSD-3-Clause |
| `setuptools` | `84.0.0` | MIT |

## External runtime and model terms

- The standalone `ffmpeg.exe` helper is absent from the inspected archive. The archive's OpenCV video-I/O component and its embedded libswscale license-string check are recorded below. No separately named OpenCV/FFmpeg notice file was found by the archive filename audit. This is an evidence gap, not a legal conclusion.
- The separately tested external helper reported FFmpeg 6.0-era build `2aec86695a`, configured with `--enable-gpl` and `libx264`; `ffmpeg -L` reports GPL version 3 or later. Users supply their own FFmpeg or use an already-installed local helper. Keep this external executable distinct from the bundled OpenCV video-I/O DLL when describing distribution contents.
- Depth model weights are downloaded on demand and are not embedded in the EXE. On 2026-10-21, the catalog's pinned model revisions were checked against their upstream model pages: Depth Anything V2 Small FP32/FP16 uses `onnx-community/depth-anything-v2-small` revision `4472b7362082ad9968fee890ca0f1e5aca36b93d`; Base uses `onnx-community/depth-anything-v2-base` revision `dd4557d492cd7b563738ac8d9ccff9094620983c`; Large uses `onnx-community/depth-anything-v2-large` revision `1fa1591c7b080e98da9655827c3a33a3972a4a83`; DPT Hybrid/MiDaS uses the Xenova ONNX conversion `Xenova/dpt-hybrid-midas` revision `8af5a62e326ba3e842759aa27e13008c1c758db5`. Hugging Face marks the Small ONNX model Apache-2.0 and Base/Large ONNX models CC-BY-NC-4.0; the original Depth Anything V2 project says Small weights are Apache-2.0 and Base/Large weights are CC-BY-NC-4.0, and a maintainer clarification says the non-commercial terms apply to those weights. The Intel upstream DPT-Hybrid model card states Apache 2.0 and the Xenova repository identifies Intel's model as its base model, but the conversion's own page does not visibly state a separate license. This evidence supports the catalog labels; it is not a legal determination for converted weights or the planned delivery model. Before release, the owner/reviewer must decide whether exposing non-commercial models is appropriate for the intended distribution, confirm the Xenova conversion terms, and recheck the pinned files and their attribution if model revisions change. Custom ONNX remains under the user's license.
- No project-level `LICENSE`, `COPYING`, `NOTICE`, or third-party notices file was present when this inventory was generated. The repository owner has since directed broad public reuse and added the root MIT `LICENSE` for original StereoLab source and documentation. That choice does not replace third-party notices or establish executable/model redistribution rights.

## Frozen archive inspection

Inspected `.runtime/mit-license-packaging-20261021-01/dist/StereoLab.exe` — 98,572,977 bytes; SHA-256 `6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45`.
Root MIT entry extraction and exact byte/hash comparison against source and sidecar: `.runtime/mit-license-packaging-20261021-01/license-package-audit.json` (PASS).
CArchive entries: 183; PYZ modules: 975; native DLL/PYD entries: 97.
Bundled OpenCV video-I/O DLL: `cv2\opencv_videoio_ffmpeg500_64.dll`; 30,876,160 bytes; SHA-256 `7aabff029d1fa47cf56ede4dbbe05dc62da3208f07cd16471d4c966a33f9aa31`; libswscale license string present: `True`; codec configuration strings: libopenh264, libvpx, libaom.
Named license/notice entries: 23 total, comprising the root MIT LICENSE and 22 third-party entries across 5 archive owners (click-8.5.0.dist-info, numpy-2.5.4.dist-info, onnx-1.23.2.dist-info, opentelemetry_api-1.45.1.dist-info, pydantic-2.14.1.dist-info). A standalone third-party notices file is absent.
Embedded build manifest fingerprint: `8e56afce8aa2240e94a07f27bc093186939dd3ed0302c72976a51baa76d9e2fe`; recomputes correctly: `True`; matches current source: `True`.
The archive notice-name scan found named license files only for the listed owners. FastAPI, Starlette, Uvicorn, python-multipart, ONNX Runtime, OpenCV, pydantic_core, and setuptools have no corresponding named license/notice files in the archive scan. Native archive components also include Python, OpenSSL, libffi, SQLite, DirectML, and Microsoft runtime DLLs without separately named notices found by this scan. These results require upstream notice and provenance review; they do not determine compliance.
The installed runtime closure lists `flatbuffers`, `mpmath`, and `sympy`, which are absent from the inspected PYZ; the archive instead contains setuptools and vendored jaraco, more_itertools, tomli, wheel, and backports modules. Treat the installed closure and frozen archive as separate inventories.

## Installer compiler

The private `v0.2.0-beta.1` setup candidate was compiled with Inno Setup 6.7.3.
The vendor asks commercial users to purchase a license and states that using
the compiler through a build script counts as use. Confirm whether the planned
build/distribution context is commercial and obtain the required license if
so; this installer smoke test does not establish compiler entitlement. See the
[vendor purchase guidance](https://jrsoftware.org/ishelp/topic_purchase.htm).

## Candidate-specific notice source review

The current 98,572,977-byte candidate has SHA-256 `6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45`. Relative to the prior candidate, the CArchive adds only the root `LICENSE`; its sole changed entry is `base_library.zip`, whose 155 members have identical names and content. All 97 native DLL/PYD entries and 22 third-party notice entries match by name and bytes. The earlier independent installed-source checks therefore carry over to those exact entries. The notices cover Click 8.5.0, NumPy 2.5.4, ONNX 1.23.2, OpenTelemetry API 1.45.1, and Pydantic 2.14.1.

The archive contains code from annotated-doc, annotated-types, AnyIO, FastAPI, h11, idna, ml_dtypes, ONNX Runtime DirectML, OpenCV, Packaging, Protobuf, pydantic_core, python-multipart, Starlette, typing_extensions, typing-inspection, Uvicorn, and setuptools without corresponding named package notices. The local `.venv-gpu` installation has the identified upstream package license files. A smallest file-preserving set of 31 additional texts covers those mapped Python/runtime components, relevant setuptools vendored components, and the CPython license. The review draft contains 38 supplementary files (37 local package/runtime copies plus one pinned-source FFmpeg notice excerpt) with 32 distinct hashes not already present in the candidate notices; it includes duplicates and a `setuptools/config/NOTICE` whose named schema files were not found in the candidate, so that entry's scope is unresolved. NumPy's existing license already contains its declared OpenBLAS, LAPACK, and GCC runtime notices; no duplicate NumPy notices are needed. These source copies are not yet assembled into an approved delivery bundle or added to the build.

All 97 archived native DLL/PYD entries were matched byte-for-byte to local package, CPython, or Codex runtime copies. This proves local byte identity only. Local notice sources were identified for CPython, ONNX Runtime, and OpenCV. The candidate's `DirectML.dll` is version `1.15.4.0` and is byte-identical to `onnxruntime/capi/DirectML.dll` from Microsoft's published `onnxruntime-directml 1.24.4` CPython 3.12 wheel; the downloaded wheel SHA matches the PyPI digest and the wheel `RECORD` verifies the DLL. The Microsoft.AI.DirectML 1.15.4 NuGet DLL has the same version and size but is a different binary. The wheel's `LICENSE` and `ThirdPartyNotices.txt` do not mention DirectML; applicability of DirectML terms/notices remains owner/reviewer work.

### Microsoft CRT and UCRT files

The candidate's 48 root CRT/API-set entries match the paths recorded in the PyInstaller Analysis TOC under `C:\Users\Hp\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\libheif\libheif\bin`; the archived and local hashes agree. Four VC runtime DLLs report PE version `14.51.36247.0`; `ucrtbase.dll` and 43 API-set DLLs report `10.0.26100.4654`. Authenticode verification of the 48 local copies returned `Valid`, with Microsoft signers. No license or redistribution document was present in that source directory. Microsoft's [VC++ redistribution guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170) and [UCRT deployment guidance](https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment?view=msvc-170) describe conditional redistribution and the UCRT local-deployment family. The candidate UCRT version matches the [Windows 11 SDK 10.0.26100.4654 catalog entry](https://learn.microsoft.com/en-us/windows/apps/windows-sdk/downloads); the SDK can be acquired through several channels, so this version match does not identify how these files were obtained. Microsoft's [Visual Studio 2026 distributable-code list](https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution) makes VC runtime distribution conditional on the terms for the actually licensed edition and qualifying unmodified redistributable files. These facts do not establish which original installer/SDK supplied the copies, the applicable agreement, or the owner's entitlement to redistribute them. The directory name does not establish that libheif itself is included.

The `14.51.36247.0` VC runtime version is from the Visual Studio 2026-era toolset. Microsoft's [compiler version table](https://learn.microsoft.com/en-gb/cpp/overview/compiler-versions?view=msvc-160) and [2026 redistribution list](https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution) show that this requires reviewing the matching edition's terms; a valid signature and an unset PE prerelease flag do not establish the source channel or entitlement. The UCRT version corresponds to a Windows 11 SDK catalog entry, but that version match does not prove acquisition from that SDK.

A targeted local provenance recheck on 2026-10-21 independently compared all 48
candidate root CRT/API-set archive entries with the Codex `native/libheif`
runtime directory: all 48 matched byte-for-byte, with no mismatches. The local
Visual Studio installation reports VC redist 14.44.35211.00, while the four
candidate VC DLLs are 14.51.36247.0; the cached 14.44 redistributable is not
their source. The installed Windows SDK redist contains names for 44 of those
entries, but its files are 10.0.26100.7705 versus the candidate UCRT/API files
at 10.0.26100.4654, with no exact hash matches. Codex runtime metadata names
libheif but does not identify the Microsoft package, SDK acquisition channel,
or applicable redistribution agreement. Microsoft signatures establish signer
identity, not redistribution entitlement; exact package/channel evidence and
owner/reviewer assessment remain open.

The candidate also contains `numpy.libs/msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll` (575,056 bytes; SHA-256 `a4c2229bdc2a2a630acdc095b4d86008e5c3e3bc7773174354f3da4f5beb9cde`), PE version `14.40.33810.0`. It matches the local NumPy package copy and has a valid Microsoft compatibility-publisher signature. Its separate provenance and applicable redistribution terms remain unverified.

### DirectML runtime

The candidate's `onnxruntime/capi/DirectML.dll` is 18,527,776 bytes, SHA-256 `b73972115320e906a49602f2027a3266622881b0d325ba685e0f165a9482a8d7`, and PE version `1.15.4.0`. It matches the local ONNX Runtime package copy and has a valid Microsoft Windows Publisher signature. Microsoft's [DirectML version history](https://learn.microsoft.com/en-us/windows/ai/directml/dml-version-history) lists version 1.15.4 as a redistributable release; the [exact NuGet license terms](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4/License) limit distribution to qualifying Windows/Xbox applications or services developed with machine-learning technologies and restrict standalone distribution. They also warn that third-party components may have separate notices. The exact Microsoft.AI.DirectML 1.15.4 NuGet package was downloaded from the [official NuGet page](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) into `.runtime/directml-package-review-20261021/Microsoft.AI.DirectML.1.15.4.nupkg` (202,292,617 bytes; SHA-256 `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9`). NuGet lists it as a 192.92 MB redistributable with no package dependencies ([package page](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4)); Microsoft's [version history](https://learn.microsoft.com/en-us/windows/ai/directml/dml-version-history) lists 1.15.4 as the redistributable for that version. The NuGet package's `bin/x64-win/DirectML.dll` is version `1.15.4.0`, 18,527,776 bytes, SHA-256 `9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1`; it differs from the candidate DLL of the same version and size. Instead, the candidate DLL byte-matches `onnxruntime/capi/DirectML.dll` from Microsoft's [official PyPI `onnxruntime-directml 1.24.4` CPython 3.12 wheel](https://pypi.org/pypi/onnxruntime-directml/1.24.4/json). The downloaded 25,111,930-byte wheel SHA-256 `f2ecb68b7b7b259d2ef3112ae760149f9b5a1e7c0fbb73d539da6250a648a614` matches PyPI's published digest; its `RECORD` verifies the embedded DLL hash and size. The wheel's `onnxruntime/LICENSE` and `ThirdPartyNotices.txt` also match their `RECORD` entries, but neither mentions DirectML. The NuGet package contains `LICENSE-CODE.txt`, `LICENSE.txt`, and `ThirdPartyNotices.txt`; its third-party notice does not mention DirectML. Full comparison report: `.runtime/directml-package-review-20261021/inspection-report.json`. The [package terms](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4/License) permit use on Windows and Xbox and describe copying/distribution in qualifying applications or services developed with machine-learning tools/frameworks, and games for those systems; they also refer to third-party notices and limit standalone distribution. The owner or qualified reviewer must assess the intended delivery model. The public DirectML source repository's license is not a substitute for reviewing the binary package terms.

### CPython native components

The candidate's `python312.dll`, OpenSSL `libssl-3.dll`/`libcrypto-3.dll`, `sqlite3.dll`, `libffi-8.dll`, and `_ssl`/`_sqlite3`/`_ctypes` extension modules match local Python 3.12.10 installation files byte-for-byte. Its OpenSSL DLLs identify version 3.0.16; CPython's tagged [`get_externals.bat`](https://raw.githubusercontent.com/python/cpython/v3.12.10/PCbuild/get_externals.bat) lists OpenSSL source 3.0.16 and binary dependency 3.0.16.2. The SQLite DLL identifies 3.49.1 and embeds source ID `873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70`, matching the [official 3.49.1 release record](https://www.sqlite.org/releaselog/3_49_1.html); SQLite's [copyright page](https://www.sqlite.org/copyright.html) states its deliverable code is public domain and does not require a license. The source ID is version evidence, not authenticated binary provenance. The candidate libffi DLL has no PE version resource or version string; the CPython recipe lists libffi 3.4.4, but that does not prove the exact DLL build inputs. The local CPython 3.12.10 aggregate `LICENSE.txt` includes PSF/history, Windows build terms, the full libffi copyright/permission/disclaimer block, and full Apache-2.0 terms. Its libffi block matches the official libffi v3.4.4 `LICENSE`; its Apache terms section matches OpenSSL 3.0.16 `LICENSE.txt`. The draft manifest now has separate OpenSSL/libffi aliases pointing to this same aggregate copy and recording source references and unverified DLL provenance. These are mapped license texts, not proof that the matched DLL bytes came from the pinned sources or package channel. The candidate has no standalone CPython aggregate or separately identified PSF/OpenSSL/libffi notice; the supplement still needs component-owner and provenance review. The candidate also has no standalone SQLite notice; the upstream project states SQLite needs no license. The ONNX and OpenTelemetry notice files contain Apache-2.0 terms also present in the local Python aggregate; normalized terms differ only in URL scheme, while the candidate files include an Apache appendix absent from the local aggregate section. The existing ONNX and OpenTelemetry Apache notices are generic and do not identify the OpenSSL DLLs; the new OpenSSL alias points to the matching CPython aggregate section, but does not authenticate DLL provenance. Exact upstream references: [CPython 3.12.10 license](https://raw.githubusercontent.com/python/cpython/v3.12.10/LICENSE), [OpenSSL 3.0.16 license](https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.16/LICENSE.txt), and [libffi 3.4.4 license](https://raw.githubusercontent.com/libffi/libffi/v3.4.4/LICENSE). The local alias records and CPython build recipes support version association but do not authenticate the exact DLL package or build inputs. Final native attribution still needs review.

### Official Python 3.12.10 package comparison

The current Python 3.12 candidate was independently compared with the official
[Windows x64 embeddable ZIP](https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip).
The 11,133,606-byte package has SHA-256
`4acbed6dd1c744b0376e3b1cf57ce906f9dc9e95e68824584c8099a63025a3c3`, matching
the exact archive checksum and download location in Python.org's
[published SPDX record](https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip.spdx.json).
Of 97 candidate native entries, 26 names overlap this ZIP and 24 are
byte-identical: Python runtime DLLs, OpenSSL DLLs, libffi, SQLite, and 18 extension
modules. This extends the earlier local-file evidence to published-package byte
identity for that subset. It does not establish the historical acquisition
channel, verify Sigstore signatures, or authenticate every upstream build input.

The two overlapping mismatches are `vcruntime140.dll` and `vcruntime140_1.dll`:
the candidate reports 14.51.36247.0, while the official ZIP has 14.42.34438.0.
The other 71 candidate native entries are outside this CPython ZIP comparison;
the Microsoft runtime provenance and redistribution review remains open.

The official `LICENSE.txt` is 36,874 bytes, SHA-256
`e502c6b880ff58d614901495a9009c136539cd0b1e2a2abb8fc00b934c203419`, and directly
matches the local aggregate and existing r4 draft copy. The SPDX record labels
libffi 3.4.4, OpenSSL 3.0.16 and SQLite 3.49.1.0 with their CPython source-dependency
URLs. These labels support component version association; they do not replace
component-level source/build and notice review. No notice payload changed.

Full evidence is `.runtime/cpython312-official-review-20261021/provenance-report.json`;
the independent coordinator result is
`output/release-review-2026-10-21/cpython312-package-comparison.json`.

### OpenCV and its FFmpeg-backed video-I/O DLL

The candidate's `cv2.pyd` and `cv2/opencv_videoio_ffmpeg500_64.dll` match the installed `opencv-python 5.0.0.93` wheel files and RECORD. The video-I/O DLL is 30,876,160 bytes, SHA-256 `7aabff029d1fa47cf56ede4dbbe05dc62da3208f07cd16471d4c966a33f9aa31`, and MD5 `a821a1135251859655090c795af05789`. The candidate MD5 matches OpenCV 5.0.0's published `FFMPEG_FILE_HASH_BIN64` for the pinned third-party binary artifact at commit `06dc20cad65dc7fcf784f70c95d46750520889a7` ([OpenCV pinned binary configuration](https://raw.githubusercontent.com/opencv/opencv/5.0.0/3rdparty/ffmpeg/ffmpeg.cmake)). This is a corroborating hash match, not a cryptographic provenance chain. Its source scripts select FFmpeg n7.1, libvpx 1.16.0, AOM 3.14.1, and OpenH264 2.5.0 headers; pinned FFmpeg component versions also match their embedded version strings ([source selections](https://raw.githubusercontent.com/opencv/opencv_3rdparty/06dc20cad65dc7fcf784f70c95d46750520889a7/ffmpeg/download_src.sh), [component versions](https://raw.githubusercontent.com/opencv/opencv_3rdparty/06dc20cad65dc7fcf784f70c95d46750520889a7/ffmpeg/ffmpeg_version.cmake), [build configuration](https://raw.githubusercontent.com/opencv/opencv_3rdparty/06dc20cad65dc7fcf784f70c95d46750520889a7/ffmpeg/make_mingw.sh)). The pinned Dockerfile installs MinGW through Ubuntu packages without pinning the exact MinGW-w64 source/runtime revision. The OpenCV build configuration contains no GPL/nonfree enable flag. The MD5 match points to the pinned OpenCV artifact but does not identify every compiled object, build input, or applicable license; it is not a legal classification or a complete source provenance chain.

Independent extraction from the exact candidate's video-I/O DLL confirms these embedded strings at uncompressed file offsets: `/usr/bin/x86_64-w64-mingw32-g++-posix` at 24,983,014; `(ver 13.0.0)` at 24,983,053; `Extra dependencies:` at 24,985,339, followed by `pthread` at 24,985,368; `./mingw-w64-libraries/winpthreads/src/rwlock.c` at 29,580,832; and `Mingw-w64 runtime failure:` at 29,600,768. A separate string scan found no `GCC RUNTIME LIBRARY EXCEPTION`, `Lockless`, or `Fabrice Bellard` string in the DLL. These absences do not identify which code objects are present. The pinned OpenCV plugin [build recipe](https://raw.githubusercontent.com/opencv/opencv_3rdparty/06dc20cad65dc7fcf784f70c95d46750520889a7/ffmpeg/build_videoio_plugin.sh) specifies `-static-libgcc -static-libstdc++` for the plugin linker and runs `ninja install/strip`; it is a build recipe, not a record of this wheel's actual build inputs or a component-level inventory of the resulting DLL. Together with the candidate strings, these are compiler/build clues, not proof that the current upstream source or recipe produced every object in this exact DLL. The candidate's NumPy license file already contains generic GCC Runtime Library Exception text (SHA-256 `a804dff0ead9fadc5293456410bcbfc32bf024be9c4513459663fb7b442d2341`), but that notice has not been mapped to this distinct plugin. The official [Winpthreads COPYING](https://raw.githubusercontent.com/mingw-w64/mingw-w64/master/mingw-w64-libraries/winpthreads/COPYING) includes mingw-w64 project terms and a separate Lockless-derived notice; the matching upstream [rwlock.c](https://raw.githubusercontent.com/mingw-w64/mingw-w64/master/mingw-w64-libraries/winpthreads/src/rwlock.c) has a mingw-w64 copyright and permission header. Candidate notice scans find no `winpthreads`, `Lockless`, or `Fabrice Bellard` text. This identifies material for review, but exact Winpthreads source revision and binary applicability remain unverified. The official [GCC libstdc++ licensing page](https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html) says the runtime exception applies to specific files bearing the relevant notice; the generic exception in NumPy's unrelated notice does not establish which objects in this distinct plugin it covers.

Neither installed OpenCV notice file is included in the candidate. The installed `LICENSE-3RD-PARTY.txt` contains the FFmpeg LGPL 2.1 text but not the separate upstream copyright/attribution preamble in [OpenCV's FFmpeg license file](https://raw.githubusercontent.com/opencv/opencv/5.0.0/3rdparty/ffmpeg/license.txt). The review draft now carries a 763-byte excerpt of source lines 0–13 alongside the full LGPL 2.1 text in the OpenCV third-party file. This maps the notice text to the pinned source but does not establish which objects are compiled into the matched DLL. The pinned OpenH264 integration is dynamically loaded; `openh264-2.5.0-win64.dll` is not in this candidate. Notice/source mapping for statically linked GCC/libstdc++ and Winpthreads-related material, the exact wheel publisher chain, and exhaustive transitive compiled-code attribution remain open. Do not infer that no GPL/nonfree material is present solely from configuration strings.

An incomplete, source-preserving review draft is at `output/release-notices-draft-2026-10-21/`. Its 22 embedded notice entries are byte-identical to the current candidate's named notice entries; it also contains 37 supplementary local package/runtime copies and one pinned-source FFmpeg notice excerpt. The 62 source aliases refer to 60 distinct draft paths because OpenSSL and libffi each have an additional component alias to the existing CPython aggregate file. The manifest records source paths, byte counts and SHA-256 values. The supplementary set has 32 distinct non-embedded hashes and includes duplicate texts. Ten vendored aliases now label eight source distributions and local metadata versions from setuptools dist-info; the candidate PYZ has the corresponding module roots, but exact frozen-code version equivalence remains unverified. The candidate also contains generated `_validate_pyproject` modules; `setuptools/config/NOTICE` names JSON schema source files that are absent as separate archive entries, so that notice's applicability to the generated validators remains open. The current candidate's package acceptance checked all static assets against source; the embedded build fingerprint covers the listed Python components but not static assets. The notice draft is not an approved delivery bundle and has not been added to the build. Remaining work includes collecting and mapping missing exact component texts from verifiable sources, resolving Microsoft runtime entitlement/provenance and complete native attribution, reviewing the intended distribution model and model terms, The project source license is MIT. No conclusion about legal sufficiency of any executable or model redistribution is made.

#### Setuptools vendor source mapping

The exact candidate PYZ contains 130 `setuptools` modules, including the
vendored import roots below. Package names and versions come from the local
setuptools `_vendor/*.dist-info/METADATA` files recorded in the original
source manifest; the table shows that each corresponding module root is
present in this candidate. It does not prove byte-for-byte source equivalence
for the embedded Python code. The candidate's `setuptools.version` module
reports `0.dev0+unknown`, so do not present the local setuptools 84.0.0 source
version as a verified embedded version.

| Source distribution | Source version | Candidate PYZ root | Notice files in draft |
| --- | --- | --- | --- |
| `backports.tarfile` | 1.2.0 | `setuptools._vendor.backports.tarfile` | `LICENSE` |
| `jaraco.context` | 6.1.0 | `setuptools._vendor.jaraco.context` | `LICENSE` |
| `jaraco.functools` | 4.4.0 | `setuptools._vendor.jaraco.functools` | `LICENSE` |
| `jaraco.text` | 4.0.0 | `setuptools._vendor.jaraco.text` | `LICENSE` |
| `more-itertools` | 10.8.0 | `setuptools._vendor.more_itertools` | `LICENSE` |
| `packaging` | 26.0 | `setuptools._vendor.packaging` | `LICENSE`, `LICENSE.APACHE`, `LICENSE.BSD` |
| `tomli` | 2.4.0 | `setuptools._vendor.tomli` | `LICENSE` |
| `wheel` | 0.46.3 | `setuptools._vendor.wheel` | `LICENSE.txt` |

`setuptools/config/_validate_pyproject` and its generated validation modules
are present in the candidate PYZ. The supplementary
`setuptools/config/_validate_pyproject/NOTICE` is in the review draft; it names
the fastjsonschema and validate-pyproject components, including generated
validator modules present in the candidate. The separate parent
`setuptools/config/NOTICE` names schema JSON files that are not separate
CArchive entries. Whether generated validator code contains material requiring
that parent schema notice still needs component-owner review, so both source
NOTICE texts remain in the draft pending final scope review.

The latest mechanical deduplication view is
`output/release-notices-deduplicated-review-2026-10-21-r4.zip`. It retains 54
byte-verified payloads and an alias row for each of the 62 manifest entries
(22 embedded and 40 supplementary aliases; OpenSSL/libffi map to sections in
the CPython aggregate, and one FFmpeg preamble excerpt maps to the pinned
OpenCV source). The aliases preserve the eight setuptools vendored
package/version labels with an explicit frozen-code version caveat. The
source-manifest SHA-256 is
`c489349490ea070051d45e4e914a8edd48fe6abba61073bac8f38257c2a953c9`; the ZIP
SHA-256 is `c593d4317079f6195f8fa085718d42ae2ad4d1227abc733541a4cc7d772b0778`.
Earlier r1-r3 ZIPs remain unchanged. The new notice excerpt is source-mapped
and hash-recorded, but native object attribution and final notice scope remain
open. The archive does not decide notice scope or legal
sufficiency and is not approved for release.

## Current artifact identity

Artifact: `D:\Users\Hp\Documents\ChatGPT\3d\.runtime\mit-license-packaging-20261021-01\dist\StereoLab.exe` — 98,572,977 bytes; SHA-256 `6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45`; embedded source fingerprint `8e56afce8aa2240e94a07f27bc093186939dd3ed0302c72976a51baa76d9e2fe`.

## Private Python 3.14.9 compatibility candidate — separate review

This candidate is distinct from the CPython 3.12 release candidate documented above and has not been promoted. Its private EXE is `.runtime/python314-compatibility-review/dist-python314/StereoLab.exe`, 99,782,837 bytes, SHA-256 `e5afd208495e9b1690fb0bd75cefb692e367195f2e9c8761060595cbf79b190e`. Its embedded build manifest matches source fingerprint `8e56afce8aa2240e94a07f27bc093186939dd3ed0302c72976a51baa76d9e2fe`.

The independent candidate provenance report confirms all 27 CPython native entries match Python.org's Python 3.14.9 embeddable ZIP byte-for-byte. That ZIP SHA-256 is `3c01d7cfc666f746a6f064f60fb2156ab0ec464d3d738ab796371a46a9bb24a2`. The candidate's DirectML DLL SHA-256 `13af15778e6e587169e5990ce8c51fd9964ca7ece8000df2a8b4f6b3fbd5f34a` matches the official `onnxruntime-directml 1.24.4` CPython 3.14 wheel (SHA-256 `51d86bb949488e572b00422f344990a4a81d982416d73b6c0e4ced2bcd423d19`), whose published digest and wheel RECORD were checked. This wheel DLL differs from the DirectML 1.15.4 NuGet DLL already reviewed for the Python 3.12 candidate. The NuGet-vs-wheel terms and notice applicability still need owner/reviewer assessment.

All 22 named license/notice archive entries were independently extracted from this EXE and matched by entry name and SHA-256 to the existing r4 aliases. This carry-forward applies only to those exact notice entries. The archive has 98 native DLL/PYD entries versus 97 in the Python 3.12 candidate: 17 added, 16 removed, 26 same-name entries changed, and 55 unchanged. The new `_zstd.pyd` is 503,520 bytes (SHA-256 `473c52fdabf4e0741d91e1a3b1cd2b4b983bd62823832fd0386cec91b6914976`). Python 3.14.9's `LICENSE.txt` is 35,407 bytes, SHA-256 `935cf13e19f8c31b497d20b05d73623431a226b230c3599bc30fa3348979bc68`; the r4 Python 3.12.10 license is 36,874 bytes, SHA-256 `e502c6b880ff58d614901495a9009c136539cd0b1e2a2abb8fc00b934c203419`. The newer CPython license includes Zstandard BSD text.

Package-source review reports FastAPI changing from 0.143.3 to 0.158.1 and Starlette from 1.8.0 to 1.9.0; both license texts are byte-identical to r4, but candidate-specific version/source aliases still need review. The build environment reports setuptools 84.2.1 while frozen `setuptools.version` is `0.dev0+unknown`; exact vendored-code versions remain unverified. Packaging metadata is 26.3 while the old setuptools vendor alias says 26.0, so refresh that mapping after identifying the frozen code source. Installed package metadata must not be treated as the frozen-code version without that mapping.

PyInstaller warning-file comparison found two newly reported keys (`excluded __main__`, `collections.abc`) and eight removed keys. Python 3.14 registers `collections.abc` as an alias to `_collections_abc`; the candidate's packaged health and generated image/video jobs imported and used the app path successfully. No other newly material warning key was identified. Build stdout/stderr were not retained, so this classification is limited to warning files and archive/runtime checks.

A separate, incomplete delta review package is `output/release-notices-deduplicated-review-2026-10-21-python314-delta.zip` (207,544 bytes; SHA-256 `2e813c2ab8fcd300a8a32bf20bb5c3c2314b9f9070c7bfe965ff66911136806`). It nests the unchanged r4 review ZIP and adds the exact Python 3.14.9 license plus a candidate delta manifest. It is not a consolidated notice source manifest, approved delivery bundle, legal review, or build input. Review remaining native notice scope, source/version aliases, Microsoft CRT/UCRT entitlement, DirectML terms, and all unresolved project/model and native attribution before redistribution.

Evidence files include `.runtime/python314-compatibility-review/python314-candidate-provenance.json`, `.runtime/python314-compatibility-review/install-report.json`, `.runtime/python314-compatibility-review/build-python314/StereoLab/warn-StereoLab.txt`, `.runtime/python314-compatibility-review/launch-smoke-20261021-3a912cf2/generated-video-smoke-report.json`, and the Python 3.14 candidate's full GPU test log. A one-second generated test-pattern video exported to 3840x1080 with four video frames and 1.002-second audio. It used OpenCV CPU visual approximation (`gpu_verified=false`); NVIDIA NVENC was only the encoder. Physical XREAL 1S playback remains unverified. A targeted Ctrl+Break run left no candidate process/listener and an empty extraction temp directory, but returned code 3; clean application exit is not claimed.
