import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import clsx from 'clsx'

interface PaginationProps {
  page: number
  totalPages: number
  onPageChange: (page: number) => void
  perPage: number
  total: number
  className?: string
}

export function Pagination({ 
  page, 
  totalPages, 
  onPageChange, 
  perPage,
  total,
  className = '' 
}: PaginationProps) {
  const startItem = (page - 1) * perPage + 1
  const endItem = Math.min(page * perPage, total)
  
  // Calculate visible page numbers
  const getPageNumbers = () => {
    const pages: (number | string)[] = []
    const maxVisible = 7
    const halfVisible = Math.floor(maxVisible / 2)
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i)
      }
    } else {
      if (page <= halfVisible + 1) {
        // Near start
        for (let i = 1; i <= maxVisible - 2; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      } else if (page >= totalPages - halfVisible) {
        // Near end
        pages.push(1)
        pages.push('...')
        for (let i = totalPages - (maxVisible - 3); i <= totalPages; i++) {
          pages.push(i)
        }
      } else {
        // Middle
        pages.push(1)
        pages.push('...')
        for (let i = page - 1; i <= page + 1; i++) {
          pages.push(i)
        }
        pages.push('...')
        pages.push(totalPages)
      }
    }
    
    return pages
  }
  
  return (
    <div className={clsx('flex flex-col sm:flex-row items-center justify-between gap-4', className)}>
      {/* Results summary */}
      <div className="text-sm text-dark-muted">
        Showing <span className="font-medium text-white">{startItem}</span> to{' '}
        <span className="font-medium text-white">{endItem}</span> of{' '}
        <span className="font-medium text-white">{total}</span> results
      </div>
      
      {/* Page controls */}
      <div className="flex items-center gap-1">
        {/* First page */}
        <button
          onClick={() => onPageChange(1)}
          disabled={page === 1}
          className={clsx(
            'p-1.5 rounded-lg transition-all',
            page === 1
              ? 'text-dark-muted cursor-not-allowed'
              : 'text-white hover:bg-dark-border hover:text-info'
          )}
          title="First page"
        >
          <ChevronsLeft className="w-4 h-4" />
        </button>
        
        {/* Previous page */}
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page === 1}
          className={clsx(
            'p-1.5 rounded-lg transition-all',
            page === 1
              ? 'text-dark-muted cursor-not-allowed'
              : 'text-white hover:bg-dark-border hover:text-info'
          )}
          title="Previous page"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        
        {/* Page numbers */}
        <div className="flex items-center gap-1 mx-2">
          {getPageNumbers().map((pageNum, idx) => (
            pageNum === '...' ? (
              <span key={`ellipsis-${idx}`} className="px-2 text-dark-muted">
                ...
              </span>
            ) : (
              <button
                key={pageNum}
                onClick={() => onPageChange(pageNum as number)}
                className={clsx(
                  'min-w-[32px] h-8 px-2 rounded-lg font-medium text-sm transition-all',
                  pageNum === page
                    ? 'bg-gradient-to-r from-info to-purple-500 text-white shadow-lg shadow-info/25'
                    : 'hover:bg-dark-border hover:text-info'
                )}
              >
                {pageNum}
              </button>
            )
          ))}
        </div>
        
        {/* Next page */}
        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page === totalPages}
          className={clsx(
            'p-1.5 rounded-lg transition-all',
            page === totalPages
              ? 'text-dark-muted cursor-not-allowed'
              : 'text-white hover:bg-dark-border hover:text-info'
          )}
          title="Next page"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        
        {/* Last page */}
        <button
          onClick={() => onPageChange(totalPages)}
          disabled={page === totalPages}
          className={clsx(
            'p-1.5 rounded-lg transition-all',
            page === totalPages
              ? 'text-dark-muted cursor-not-allowed'
              : 'text-white hover:bg-dark-border hover:text-info'
          )}
          title="Last page"
        >
          <ChevronsRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}