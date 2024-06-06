import multiprocessing as mp
import time
import typing as t
from threading import Lock

import numpy as np

from helper.ent_extraction import extract_entities
from helper.models import Task
from settings import *
from helper.texts import sentence_splitter
from helper.utils import query_embedder

T = t.TypeVar("T")


def task_wrapper_extract_entities(task: Task, verify_lock: Lock):
    start = time.perf_counter()
    q_embeds = np.array(query_embedder(task.type))

    try:
        task.extracted_ents = extract_entities(
            verify=task.verify,
            sentences=sentence_splitter(task.text),
            temperature=LLM_TEMPERATURE,
            entity_type=task.type,  # type: ignore
            q_embeds=q_embeds,  # type: ignore
            verify_lock=verify_lock,
        )
    except Exception as e:
        task.extracted_ents = set()
        lg.debug("Error: {}\n{}".format(task.paper.id, e))

    task.time_elapsed = time.perf_counter() - start
    task.pending = False
    return task


def forker(tasks: list[T], function: t.Callable[[T], T]):
    start = time.perf_counter()

    done__tasks: list[T] = []

    lg.info(f"Total tasks: {len(tasks)}.")

    with mp.Pool(processes=5, maxtasksperchild=2) as e:
        start = time.perf_counter()
        results = e.imap_unordered(function, tasks)
        lg.debug(f"Mapped processes! {time.perf_counter() - start}s")
        for result in results:
            done__tasks.append(result)

    lg.info("Done!")

    return (done__tasks, time.perf_counter() - start)
