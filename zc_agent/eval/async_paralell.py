from tqdm.asyncio import tqdm_asyncio
import asyncio


async def map_progress(
            sequence,
            function,
            desc="Processing",
            parallelism=6
        ):
    semaphore = asyncio.Semaphore(parallelism)

    async def sem_task(item):
        async with semaphore:
            return await function(item)

    tasks = [sem_task(item) for item in sequence]

    results = []

    wrapped_tasks = tqdm_asyncio.as_completed(
        tasks,
        desc=desc,
        total=len(tasks)
    )

    for result in wrapped_tasks:
        res = await result
        results.append(res)

    return results

