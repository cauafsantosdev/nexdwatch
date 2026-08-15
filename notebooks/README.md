# Historical collection notebooks

The notebooks in this directory preserve early dataset-acquisition provenance. They
use the retired `scrapxd` integration and are not current ingestion or development
instructions.

Production username synchronization now uses `letterboxdpy` through the Celery
`profile_sync` worker. Official Letterboxd export ZIP ingestion is the supported
offline fallback. Keep these notebooks unchanged until the provenance of the local
historical CSVs has been documented or archived elsewhere.
