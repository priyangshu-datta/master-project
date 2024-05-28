import multiprocessing as mp
import time
import typing as t

import loguru as lg

from CONSTANTS import LLM_TEMPERATURE
from ent_extraction import extract_entities, query_embedder
from models import Task
from texts import chunker
import numpy as np

T = t.TypeVar("T")


def task_wrapper_extract_entities(task: Task):
    start = time.perf_counter()
    q_embeds = query_embedder(task.type)
    lg.logger.debug(f"Query embeddings calculated! {time.perf_counter() - start}s")
    task.extracted_ents = extract_entities(
        verify=task.verify,
        chunks=chunker(task.text, task.paper.id),
        temperature=LLM_TEMPERATURE,
        entity_type=task.type,  # type: ignore
        q_embeds=np.array(q_embeds.get("embeddings")),  # type: ignore
    )
    task.time_elapsed = time.perf_counter() - start
    task.pending = False

    # lg.logger.debug(task)

    # q.put(task)
    # return task


def forker(tasks: list[T], function: t.Callable[[T], T]):
    start = time.perf_counter()

    done__tasks: list[T] = []

    lg.logger.info(f"Total tasks: {len(tasks)}.")

    # processes: t.List[mp.Process] = []
    # queue: mp.Queue = mp.Queue()

    # for task in tasks:
    #     processes.append(mp.Process(target=function, args=(task, queue), ))

    # for p in processes:
    #     p.start()

    # for p in processes:
    #     p.terminate()
    #     p.join()

    # for _ in range(queue.qsize()):
    #     done__tasks.append(queue.get())

    with mp.Pool(processes=5, maxtasksperchild=2) as e:
        start = time.perf_counter()
        results = e.imap_unordered(function, tasks[:4])
        lg.logger.debug(f"Mapped processes! {time.perf_counter() - start}s")
        for result in results:
            done__tasks.append(result)

    lg.logger.info("Done!")

    return (done__tasks, time.perf_counter() - start)
